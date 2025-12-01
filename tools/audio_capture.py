"""
Audio Capture Module for MinutesX

Captures system audio (loopback) to listen to Google Meet and other meeting platforms.
Supports real-time audio streaming for live transcription.
"""
import asyncio
import io
import queue
import threading
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Generator
import time

from observability.logger import get_logger


logger = get_logger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of captured audio."""
    data: bytes
    sample_rate: int
    channels: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0
    
    def to_wav_bytes(self) -> bytes:
        """Convert audio chunk to WAV format bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)  # 16-bit audio
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.data)
        return buffer.getvalue()


class AudioCaptureBase(ABC):
    """Base class for audio capture implementations."""
    
    @abstractmethod
    def start(self) -> None:
        """Start audio capture."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop audio capture."""
        pass
    
    @abstractmethod
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """Get the next audio chunk."""
        pass
    
    @abstractmethod
    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        pass


class SystemAudioCapture(AudioCaptureBase):
    """
    Captures system audio using WASAPI loopback (Windows) or PulseAudio (Linux).
    
    This allows capturing audio playing through speakers/headphones,
    perfect for capturing Google Meet audio.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 5.0,  # seconds per chunk
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.device_index = device_index
        
        self._audio_queue: queue.Queue[AudioChunk] = queue.Queue()
        self._is_capturing = False
        self._capture_thread: Optional[threading.Thread] = None
        self._stream = None
        self._audio_interface = None
        
        logger.info(f"SystemAudioCapture initialized: {sample_rate}Hz, {channels}ch, {chunk_duration}s chunks")
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """List available audio devices."""
        devices = []
        try:
            import sounddevice as sd
            device_list = sd.query_devices()
            for i, dev in enumerate(device_list):
                devices.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                    "is_loopback": "loopback" in dev["name"].lower() or "stereo mix" in dev["name"].lower(),
                })
        except ImportError:
            logger.warning("sounddevice not installed, trying pyaudio")
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    devices.append({
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["maxInputChannels"],
                        "sample_rate": int(dev["defaultSampleRate"]),
                        "is_loopback": "loopback" in dev["name"].lower() or "stereo mix" in dev["name"].lower(),
                    })
                p.terminate()
            except ImportError:
                logger.error("No audio library available")
        
        return devices
    
    def find_loopback_device(self) -> Optional[int]:
        """Find a loopback/stereo mix device for system audio capture."""
        devices = self.list_devices()
        
        # Priority: WASAPI Loopback > Stereo Mix > any loopback
        for dev in devices:
            if "wasapi" in dev["name"].lower() and "loopback" in dev["name"].lower():
                return dev["index"]
        
        for dev in devices:
            if "stereo mix" in dev["name"].lower():
                return dev["index"]
        
        for dev in devices:
            if dev.get("is_loopback"):
                return dev["index"]
        
        logger.warning("No loopback device found. Enable 'Stereo Mix' in Windows sound settings.")
        return None
    
    def start(self) -> None:
        """Start capturing system audio."""
        if self._is_capturing:
            logger.warning("Already capturing")
            return
        
        # Find device if not specified
        device_idx = self.device_index or self.find_loopback_device()
        
        if device_idx is None:
            # Fall back to default microphone for testing
            logger.warning("No loopback device found, using default input device")
        
        self._is_capturing = True
        self._capture_thread = threading.Thread(target=self._capture_loop, args=(device_idx,))
        self._capture_thread.daemon = True
        self._capture_thread.start()
        
        logger.info(f"Audio capture started (device: {device_idx})")
    
    def _capture_loop(self, device_index: Optional[int]) -> None:
        """Main capture loop running in separate thread."""
        try:
            import sounddevice as sd
            import numpy as np
            
            chunk_samples = int(self.sample_rate * self.chunk_duration)
            audio_buffer = []
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                audio_buffer.append(indata.copy())
            
            # Open input stream
            with sd.InputStream(
                device=device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype='int16',
                callback=audio_callback,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            ):
                while self._is_capturing:
                    # Wait for enough audio data
                    time.sleep(self.chunk_duration)
                    
                    if audio_buffer:
                        # Combine buffered audio
                        combined = np.concatenate(audio_buffer)
                        audio_buffer.clear()
                        
                        # Create audio chunk
                        chunk = AudioChunk(
                            data=combined.tobytes(),
                            sample_rate=self.sample_rate,
                            channels=self.channels,
                            duration_seconds=len(combined) / self.sample_rate,
                        )
                        
                        self._audio_queue.put(chunk)
                        logger.debug(f"Audio chunk captured: {chunk.duration_seconds:.2f}s")
                        
        except ImportError:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
            self._capture_with_pyaudio(device_index)
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            self._is_capturing = False
    
    def _capture_with_pyaudio(self, device_index: Optional[int]) -> None:
        """Fallback capture using PyAudio."""
        try:
            import pyaudio
            import numpy as np
            
            p = pyaudio.PyAudio()
            chunk_samples = int(self.sample_rate * 0.1)  # 100ms chunks
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_samples,
            )
            
            audio_buffer = []
            samples_needed = int(self.sample_rate * self.chunk_duration)
            
            while self._is_capturing:
                data = stream.read(chunk_samples, exception_on_overflow=False)
                audio_buffer.append(data)
                
                total_samples = sum(len(d) // 2 for d in audio_buffer)
                
                if total_samples >= samples_needed:
                    combined = b''.join(audio_buffer)
                    audio_buffer.clear()
                    
                    chunk = AudioChunk(
                        data=combined,
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        duration_seconds=len(combined) / (self.sample_rate * 2),
                    )
                    
                    self._audio_queue.put(chunk)
                    logger.debug(f"Audio chunk captured: {chunk.duration_seconds:.2f}s")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except ImportError:
            logger.error("PyAudio not installed. Run: pip install pyaudio")
        except Exception as e:
            logger.error(f"PyAudio capture error: {e}")
            self._is_capturing = False
    
    def stop(self) -> None:
        """Stop audio capture."""
        self._is_capturing = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        logger.info("Audio capture stopped")
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """Get the next audio chunk from the queue."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        return self._is_capturing
    
    def stream_audio(self) -> Generator[AudioChunk, None, None]:
        """Generator that yields audio chunks as they're captured."""
        while self._is_capturing:
            chunk = self.get_audio_chunk(timeout=self.chunk_duration + 1.0)
            if chunk:
                yield chunk


class MicrophoneCapture(AudioCaptureBase):
    """
    Captures audio from microphone.
    
    Useful for capturing your voice in a meeting.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 5.0,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.device_index = device_index
        
        self._audio_queue: queue.Queue[AudioChunk] = queue.Queue()
        self._is_capturing = False
        self._capture_thread: Optional[threading.Thread] = None
        
        logger.info("MicrophoneCapture initialized")
    
    def start(self) -> None:
        """Start microphone capture."""
        if self._is_capturing:
            return
        
        self._is_capturing = True
        self._capture_thread = threading.Thread(target=self._capture_loop)
        self._capture_thread.daemon = True
        self._capture_thread.start()
        
        logger.info("Microphone capture started")
    
    def _capture_loop(self) -> None:
        """Capture loop for microphone."""
        try:
            import sounddevice as sd
            import numpy as np
            
            chunk_samples = int(self.sample_rate * self.chunk_duration)
            
            while self._is_capturing:
                # Record audio chunk
                recording = sd.rec(
                    chunk_samples,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='int16',
                    device=self.device_index,
                )
                sd.wait()
                
                chunk = AudioChunk(
                    data=recording.tobytes(),
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    duration_seconds=self.chunk_duration,
                )
                
                self._audio_queue.put(chunk)
                
        except Exception as e:
            logger.error(f"Microphone capture error: {e}")
            self._is_capturing = False
    
    def stop(self) -> None:
        """Stop microphone capture."""
        self._is_capturing = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        logger.info("Microphone capture stopped")
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """Get the next audio chunk."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        return self._is_capturing


class AudioFileCapture(AudioCaptureBase):
    """
    Captures audio from a file.
    
    Useful for processing recorded meetings.
    """
    
    def __init__(
        self,
        file_path: str,
        chunk_duration: float = 30.0,  # Longer chunks for file processing
    ):
        self.file_path = Path(file_path)
        self.chunk_duration = chunk_duration
        
        self._audio_queue: queue.Queue[AudioChunk] = queue.Queue()
        self._is_capturing = False
        self._process_thread: Optional[threading.Thread] = None
        
        logger.info(f"AudioFileCapture initialized: {file_path}")
    
    def start(self) -> None:
        """Start processing the audio file."""
        if self._is_capturing:
            return
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")
        
        self._is_capturing = True
        self._process_thread = threading.Thread(target=self._process_file)
        self._process_thread.daemon = True
        self._process_thread.start()
        
        logger.info("Audio file processing started")
    
    def _process_file(self) -> None:
        """Process audio file and create chunks."""
        try:
            with wave.open(str(self.file_path), 'rb') as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                
                chunk_frames = int(sample_rate * self.chunk_duration)
                
                while self._is_capturing:
                    frames = wav.readframes(chunk_frames)
                    if not frames:
                        break
                    
                    chunk = AudioChunk(
                        data=frames,
                        sample_rate=sample_rate,
                        channels=channels,
                        duration_seconds=len(frames) / (sample_rate * channels * sample_width),
                    )
                    
                    self._audio_queue.put(chunk)
                    logger.debug(f"Audio chunk processed: {chunk.duration_seconds:.2f}s")
            
            self._is_capturing = False
            logger.info("Audio file processing completed")
            
        except Exception as e:
            logger.error(f"Audio file processing error: {e}")
            self._is_capturing = False
    
    def stop(self) -> None:
        """Stop processing."""
        self._is_capturing = False
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """Get the next audio chunk."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_capturing(self) -> bool:
        """Check if currently processing."""
        return self._is_capturing or not self._audio_queue.empty()


def create_audio_capture(
    source: str = "system",
    **kwargs,
) -> AudioCaptureBase:
    """
    Factory function to create appropriate audio capture.
    
    Args:
        source: Audio source type - "system", "microphone", or file path
        **kwargs: Additional arguments for the capture class
        
    Returns:
        AudioCaptureBase instance
    """
    if source == "system":
        return SystemAudioCapture(**kwargs)
    elif source == "microphone":
        return MicrophoneCapture(**kwargs)
    elif Path(source).exists():
        return AudioFileCapture(source, **kwargs)
    else:
        raise ValueError(f"Unknown audio source: {source}")
