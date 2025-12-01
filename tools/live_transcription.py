"""
Live Transcription Service for MinutesX

Uses Google Gemini 2.5 Flash for real-time speech-to-text transcription.
Processes audio chunks and generates live captions/transcripts.
"""
import asyncio
import base64
import io
import json
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional

import google.generativeai as genai

from config import config
from observability.logger import get_logger
from observability.metrics import metrics_collector
from tools.audio_capture import AudioChunk, AudioCaptureBase


logger = get_logger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a transcribed segment of audio."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    speaker: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_final: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "speaker": self.speaker,
            "timestamp": self.timestamp.isoformat(),
            "is_final": self.is_final,
        }


@dataclass
class LiveCaption:
    """Represents a live caption for display."""
    text: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    speaker: Optional[str] = None
    
    def __str__(self) -> str:
        speaker_prefix = f"[{self.speaker}] " if self.speaker else ""
        return f"{speaker_prefix}{self.text}"


class LiveTranscriptionService:
    """
    Real-time transcription service using Gemini 2.5 Flash.
    
    Features:
    - Live audio transcription
    - Speaker identification
    - Real-time caption generation
    - Transcript accumulation
    """
    
    TRANSCRIPTION_PROMPT = """You are a professional transcriptionist. Listen to this audio and provide an accurate transcription.

Instructions:
1. Transcribe the speech accurately, including all spoken words
2. Use proper punctuation and capitalization
3. If you can identify different speakers, label them as Speaker 1, Speaker 2, etc.
4. Include filler words like "um", "uh" only if they seem intentional
5. If audio is unclear, indicate with [inaudible]
6. Focus on the actual spoken content, ignore background noise descriptions

Return ONLY the transcription text, no additional commentary. If there is no speech in the audio, return "[silence]"."""

    CAPTION_PROMPT = """Convert this transcript segment into a clean, readable caption suitable for live display.

TRANSCRIPT:
{transcript}

Rules:
1. Keep it concise (max 100 characters if possible)
2. Remove filler words (um, uh, like, you know)
3. Fix grammar naturally
4. Preserve the core meaning
5. Make it suitable for real-time display

Return ONLY the cleaned caption text."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name or config.gemini.model
        api_key = api_key or config.gemini.api_key
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for transcription")
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.1,  # Low temperature for accuracy
                "max_output_tokens": 2048,
            }
        )
        
        self._transcript_segments: List[TranscriptSegment] = []
        self._caption_queue: queue.Queue[LiveCaption] = queue.Queue()
        self._is_running = False
        self._total_duration = 0.0
        self._callbacks: List[Callable[[TranscriptSegment], None]] = []
        self._caption_callbacks: List[Callable[[LiveCaption], None]] = []
        
        logger.info(f"LiveTranscriptionService initialized with model: {self.model_name}")
    
    def add_transcript_callback(self, callback: Callable[[TranscriptSegment], None]):
        """Add a callback to be called when new transcript is available."""
        self._callbacks.append(callback)
    
    def add_caption_callback(self, callback: Callable[[LiveCaption], None]):
        """Add a callback for live captions."""
        self._caption_callbacks.append(callback)
    
    def transcribe_audio_chunk(
        self,
        audio_chunk: AudioChunk,
        include_speaker: bool = True,
    ) -> Optional[TranscriptSegment]:
        """
        Transcribe a single audio chunk using Gemini.
        
        Args:
            audio_chunk: The audio chunk to transcribe
            include_speaker: Whether to attempt speaker identification
            
        Returns:
            TranscriptSegment with transcribed text
        """
        try:
            # Convert audio to base64 for Gemini
            wav_bytes = audio_chunk.to_wav_bytes()
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            # Create multimodal content with audio
            content = [
                self.TRANSCRIPTION_PROMPT,
                {
                    "mime_type": "audio/wav",
                    "data": audio_base64,
                }
            ]
            
            # Generate transcription
            response = self.model.generate_content(content)
            transcript_text = response.text.strip()
            
            # Skip if silence
            if transcript_text == "[silence]" or not transcript_text:
                return None
            
            # Create segment
            start_time = self._total_duration
            end_time = start_time + audio_chunk.duration_seconds
            self._total_duration = end_time
            
            segment = TranscriptSegment(
                text=transcript_text,
                start_time=start_time,
                end_time=end_time,
                confidence=0.9,  # Gemini doesn't return confidence, using default
            )
            
            # Extract speaker if mentioned
            if include_speaker and "Speaker" in transcript_text:
                segment.speaker = self._extract_speaker(transcript_text)
            
            self._transcript_segments.append(segment)
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(segment)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # Generate and notify caption
            caption = self._generate_caption(segment)
            if caption:
                for callback in self._caption_callbacks:
                    try:
                        callback(caption)
                    except Exception as e:
                        logger.error(f"Caption callback error: {e}")
            
            logger.debug(f"Transcribed: {transcript_text[:100]}...")
            return segment
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def _extract_speaker(self, text: str) -> Optional[str]:
        """Extract speaker identifier from transcript."""
        import re
        match = re.search(r'(Speaker \d+)', text)
        return match.group(1) if match else None
    
    def _generate_caption(self, segment: TranscriptSegment) -> Optional[LiveCaption]:
        """Generate a clean caption from transcript segment."""
        try:
            prompt = self.CAPTION_PROMPT.format(transcript=segment.text)
            response = self.model.generate_content(prompt)
            caption_text = response.text.strip()
            
            if caption_text:
                caption = LiveCaption(
                    text=caption_text,
                    speaker=segment.speaker,
                )
                self._caption_queue.put(caption)
                return caption
                
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            # Fallback: use transcript text directly
            return LiveCaption(text=segment.text[:100], speaker=segment.speaker)
        
        return None
    
    async def start_live_transcription(
        self,
        audio_capture: AudioCaptureBase,
        callback: Optional[Callable[[TranscriptSegment], None]] = None,
    ):
        """
        Start live transcription from an audio capture source.
        
        Args:
            audio_capture: Audio capture instance
            callback: Optional callback for each transcript segment
        """
        if callback:
            self.add_transcript_callback(callback)
        
        self._is_running = True
        logger.info("Starting live transcription...")
        
        # Start audio capture
        audio_capture.start()
        
        try:
            while self._is_running and audio_capture.is_capturing():
                chunk = audio_capture.get_audio_chunk(timeout=10.0)
                
                if chunk:
                    # Run transcription in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    segment = await loop.run_in_executor(
                        None,
                        self.transcribe_audio_chunk,
                        chunk,
                    )
                    
                    if segment:
                        metrics_collector.record_transcription_chunk(
                            chunk.duration_seconds,
                            len(segment.text.split()),
                        )
                else:
                    # No audio, brief pause
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("Live transcription cancelled")
        except Exception as e:
            logger.error(f"Live transcription error: {e}")
        finally:
            audio_capture.stop()
            self._is_running = False
            logger.info("Live transcription stopped")
    
    def stop(self):
        """Stop live transcription."""
        self._is_running = False
    
    def get_full_transcript(self) -> str:
        """Get the complete accumulated transcript."""
        return "\n".join(seg.text for seg in self._transcript_segments)
    
    def get_transcript_segments(self) -> List[TranscriptSegment]:
        """Get all transcript segments."""
        return list(self._transcript_segments)
    
    def get_captions(self, limit: int = 10) -> List[LiveCaption]:
        """Get recent captions."""
        captions = []
        while not self._caption_queue.empty() and len(captions) < limit:
            try:
                captions.append(self._caption_queue.get_nowait())
            except queue.Empty:
                break
        return captions
    
    def clear(self):
        """Clear all accumulated transcripts and captions."""
        self._transcript_segments.clear()
        self._total_duration = 0.0
        while not self._caption_queue.empty():
            try:
                self._caption_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_transcript_with_timestamps(self) -> str:
        """Get transcript with timestamps for each segment."""
        lines = []
        for seg in self._transcript_segments:
            timestamp = f"[{self._format_time(seg.start_time)} - {self._format_time(seg.end_time)}]"
            speaker = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"{timestamp} {speaker}{seg.text}")
        return "\n".join(lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


class TranscriptionProcessor:
    """
    Processes accumulated transcripts for meeting notes.
    
    Works with the agent pipeline to generate summaries,
    action items, and captions.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=config.gemini.model,
            generation_config={
                "temperature": config.gemini.temperature,
                "max_output_tokens": config.gemini.max_tokens,
            }
        )
        logger.info("TranscriptionProcessor initialized")
    
    def process_transcript(
        self,
        transcript: str,
        participants: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a complete transcript through the meeting notes pipeline.
        
        This is a simplified direct processing without the full agent system.
        For full agent processing, use MeetingOrchestratorAgent.
        """
        from agents.orchestrator_agent import MeetingOrchestratorAgent
        import uuid
        
        orchestrator = MeetingOrchestratorAgent()
        meeting_id = str(uuid.uuid4())
        
        result = orchestrator.process_meeting(
            meeting_id=meeting_id,
            transcript=transcript,
            participants=participants,
        )
        
        return result.to_dict()
    
    def generate_quick_summary(self, transcript: str) -> str:
        """Generate a quick summary without full pipeline."""
        prompt = f"""Provide a brief 3-sentence summary of this meeting:

{transcript[:15000]}

Summary:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Quick summary failed: {e}")
            return "Summary could not be generated."
    
    def extract_key_points(self, transcript: str) -> List[str]:
        """Extract key points from transcript."""
        prompt = f"""Extract the 5-7 most important points from this meeting transcript.
Return as a numbered list.

Transcript:
{transcript[:15000]}

Key Points:"""
        
        try:
            response = self.model.generate_content(prompt)
            lines = response.text.strip().split('\n')
            return [line.strip() for line in lines if line.strip()]
        except Exception as e:
            logger.error(f"Key points extraction failed: {e}")
            return []
    
    def identify_action_items(self, transcript: str) -> List[Dict[str, str]]:
        """Quick action item extraction."""
        prompt = f"""Extract action items from this meeting. For each, provide:
- Task description
- Who is responsible (if mentioned)
- Due date (if mentioned)

Return as JSON array:
[{{"task": "...", "owner": "...", "due": "..."}}]

Transcript:
{transcript[:15000]}

Action Items (JSON only):"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            # Clean up response
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            logger.error(f"Action item extraction failed: {e}")
            return []
