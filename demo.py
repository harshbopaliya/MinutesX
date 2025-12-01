"""
MinutesX Demo - Google Meet Live Transcription & Notes
=======================================================

This demo captures system audio (from Google Meet), transcribes it using
Gemini 2.5 Flash, and generates meeting notes with captions, summaries,
and action items.

Usage:
    1. Set your GOOGLE_API_KEY in .env file
    2. Start a Google Meet call
    3. Run: python demo.py
    4. Press Ctrl+C to stop and get meeting summary

Requirements:
    pip install google-generativeai sounddevice numpy python-dotenv rich
"""

import os
import sys
import time
import json
import queue
import threading
import base64
import io
import wave
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Rich for beautiful console output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

console = Console()

# Check for API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    console.print("[red]ERROR: GOOGLE_API_KEY not found![/red]")
    console.print("[yellow]Please set it in your .env file:[/yellow]")
    console.print("GOOGLE_API_KEY=your_api_key_here")
    console.print("\nGet your key at: https://aistudio.google.com/apikey")
    sys.exit(1)

# Import Google Generative AI
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# Try to import audio libraries
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    console.print("[yellow]Warning: sounddevice/numpy not installed.[/yellow]")
    console.print("[yellow]Run: pip install sounddevice numpy[/yellow]")


class MeetingRecorder:
    """Records system audio for Google Meet capture."""
    
    def __init__(self, sample_rate=16000, channels=1, chunk_seconds=10):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_seconds = chunk_seconds
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recorded_audio = []
        
    def list_devices(self):
        """List available audio devices."""
        if not AUDIO_AVAILABLE:
            return []
        
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'is_loopback': 'loopback' in dev['name'].lower() or 'stereo mix' in dev['name'].lower()
                })
        return devices
    
    def find_best_device(self):
        """Find the best device for capturing system audio."""
        devices = self.list_devices()
        
        # Priority: Loopback > Stereo Mix > Default
        for dev in devices:
            if 'loopback' in dev['name'].lower():
                return dev['index']
        
        for dev in devices:
            if 'stereo mix' in dev['name'].lower():
                return dev['index']
        
        # Return default input device
        return None
    
    def start(self, device_index=None):
        """Start recording audio."""
        if not AUDIO_AVAILABLE:
            console.print("[red]Audio capture not available.[/red]")
            return False
        
        self.is_recording = True
        self.recorded_audio = []
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                console.print(f"[yellow]Audio status: {status}[/yellow]")
            if self.is_recording:
                self.audio_queue.put(indata.copy())
        
        try:
            self.stream = sd.InputStream(
                device=device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype='int16',
                callback=audio_callback,
                blocksize=int(self.sample_rate * 0.5)  # 500ms blocks
            )
            self.stream.start()
            console.print("[green]✓ Audio recording started[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to start audio: {e}[/red]")
            self.is_recording = False
            return False
    
    def get_chunk(self, timeout=None):
        """Get a chunk of recorded audio."""
        if not self.is_recording:
            return None
        
        chunks = []
        samples_needed = int(self.sample_rate * self.chunk_seconds)
        samples_collected = 0
        
        start_time = time.time()
        timeout = timeout or (self.chunk_seconds + 2)
        
        while samples_collected < samples_needed:
            if time.time() - start_time > timeout:
                break
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                chunks.append(chunk)
                samples_collected += len(chunk)
            except queue.Empty:
                continue
        
        if chunks:
            audio_data = np.concatenate(chunks)
            self.recorded_audio.append(audio_data)
            return audio_data
        return None
    
    def stop(self):
        """Stop recording."""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        console.print("[yellow]Audio recording stopped[/yellow]")
    
    def get_all_audio(self):
        """Get all recorded audio as one array."""
        if self.recorded_audio:
            return np.concatenate(self.recorded_audio)
        return None
    
    def audio_to_wav_bytes(self, audio_data):
        """Convert audio numpy array to WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_data.tobytes())
        return buffer.getvalue()


class GeminiTranscriber:
    """Transcribes audio using Gemini 2.5 Flash."""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.transcript_parts = []
        
    def transcribe_audio(self, wav_bytes):
        """Transcribe audio using Gemini's multimodal capabilities."""
        try:
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            response = self.model.generate_content([
                "Transcribe this audio accurately. Return ONLY the spoken text, nothing else. If there's no speech, return [silence].",
                {"mime_type": "audio/wav", "data": audio_b64}
            ])
            
            text = response.text.strip()
            if text and text != "[silence]":
                self.transcript_parts.append(text)
                return text
            return None
            
        except Exception as e:
            console.print(f"[red]Transcription error: {e}[/red]")
            return None
    
    def get_full_transcript(self):
        """Get the complete transcript."""
        return "\n".join(self.transcript_parts)


class MeetingAnalyzer:
    """Analyzes meeting transcript using Gemini 2.5 Flash."""
    
    def __init__(self):
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config={"temperature": 0.3, "max_output_tokens": 4096}
        )
    
    def generate_caption(self, transcript):
        """Generate a one-line caption for the meeting."""
        prompt = f"""Create a one-line caption (max 100 characters) summarizing this meeting:

{transcript[:5000]}

Return ONLY the caption text."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Meeting on {datetime.now().strftime('%B %d, %Y')}"
    
    def generate_summary(self, transcript):
        """Generate meeting summary with key points."""
        prompt = f"""Analyze this meeting transcript and provide a summary.

TRANSCRIPT:
{transcript[:15000]}

Return a JSON object with this structure:
{{
    "executive_summary": "3-sentence executive summary",
    "key_points": ["point 1", "point 2", "point 3", "..."],
    "decisions": ["decision 1", "decision 2"],
    "topics_discussed": ["topic 1", "topic 2"]
}}

Return ONLY valid JSON."""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            console.print(f"[yellow]Summary parsing issue: {e}[/yellow]")
            return {
                "executive_summary": "Meeting summary could not be generated.",
                "key_points": [],
                "decisions": [],
                "topics_discussed": []
            }
    
    def extract_action_items(self, transcript):
        """Extract action items from transcript."""
        prompt = f"""Extract action items from this meeting transcript.

TRANSCRIPT:
{transcript[:15000]}

Return a JSON array:
[
    {{"task": "description", "owner": "person name or Unassigned", "priority": "high/medium/low"}}
]

Return ONLY valid JSON array."""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            console.print(f"[yellow]Action items parsing issue: {e}[/yellow]")
            return []
    
    def generate_live_caption(self, text):
        """Generate a clean caption from raw transcript text."""
        prompt = f"""Clean up this transcript segment for display as a live caption.
Remove filler words, fix grammar, keep it concise (max 80 chars).

Text: {text}

Return ONLY the cleaned caption."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()[:100]
        except:
            return text[:100]


def save_meeting_notes(transcript, caption, summary, action_items, output_dir="./meeting_notes"):
    """Save meeting notes to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save transcript
    transcript_file = output_path / f"meeting_{timestamp}_transcript.txt"
    transcript_file.write_text(transcript, encoding='utf-8')
    
    # Save markdown notes
    md_content = f"""# Meeting Notes
**Date:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

## Caption
{caption}

## Executive Summary
{summary.get('executive_summary', 'N/A')}

## Key Points
"""
    for point in summary.get('key_points', []):
        md_content += f"- {point}\n"
    
    md_content += "\n## Decisions Made\n"
    for decision in summary.get('decisions', []):
        md_content += f"- {decision}\n"
    
    md_content += "\n## Action Items\n"
    for item in action_items:
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.get('priority', 'medium'), "⚪")
        md_content += f"- {priority_emoji} **{item.get('task', '')}** - {item.get('owner', 'Unassigned')}\n"
    
    md_content += "\n## Topics Discussed\n"
    md_content += ", ".join(summary.get('topics_discussed', ['General discussion']))
    
    notes_file = output_path / f"meeting_{timestamp}_notes.md"
    notes_file.write_text(md_content, encoding='utf-8')
    
    # Save JSON
    json_data = {
        "timestamp": timestamp,
        "caption": caption,
        "summary": summary,
        "action_items": action_items,
        "transcript": transcript
    }
    json_file = output_path / f"meeting_{timestamp}_data.json"
    json_file.write_text(json.dumps(json_data, indent=2), encoding='utf-8')
    
    return transcript_file, notes_file, json_file


def display_results(caption, summary, action_items):
    """Display meeting results in console."""
    console.print("\n")
    console.print(Panel(f"[bold]{caption}[/bold]", title="📝 Meeting Caption", border_style="cyan"))
    
    if summary.get('executive_summary'):
        console.print(Panel(summary['executive_summary'], title="📋 Executive Summary", border_style="green"))
    
    if summary.get('key_points'):
        console.print("\n[bold]📌 Key Points:[/bold]")
        for point in summary['key_points'][:7]:
            console.print(f"  • {point}")
    
    if summary.get('decisions'):
        console.print("\n[bold]🎯 Decisions Made:[/bold]")
        for decision in summary['decisions'][:5]:
            console.print(f"  • {decision}")
    
    if action_items:
        table = Table(title="✅ Action Items", show_header=True, header_style="bold magenta")
        table.add_column("Priority", width=8)
        table.add_column("Task", width=50)
        table.add_column("Owner", width=15)
        
        for item in action_items[:10]:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.get('priority', 'medium'), "⚪")
            table.add_row(priority_emoji, item.get('task', '')[:50], item.get('owner', 'Unassigned'))
        
        console.print(table)


def run_demo():
    """Run the MinutesX demo."""
    console.print(Panel.fit(
        "[bold cyan]MinutesX - Google Meet Live Transcription Demo[/bold cyan]\n\n"
        "This demo will:\n"
        "1. Capture audio from your system (Google Meet)\n"
        "2. Transcribe speech in real-time using Gemini 2.5 Flash\n"
        "3. Generate meeting notes, captions, and action items\n\n"
        "[yellow]Press Ctrl+C to stop recording and generate notes[/yellow]",
        title="🎯 MinutesX Demo"
    ))
    
    if not AUDIO_AVAILABLE:
        console.print("\n[red]Audio capture not available. Running in transcript-only mode.[/red]")
        console.print("[yellow]To enable audio capture: pip install sounddevice numpy[/yellow]\n")
        run_transcript_demo()
        return
    
    # Initialize components
    recorder = MeetingRecorder(chunk_seconds=10)
    transcriber = GeminiTranscriber()
    analyzer = MeetingAnalyzer()
    
    # List and select audio device
    console.print("\n[cyan]Available Audio Devices:[/cyan]")
    devices = recorder.list_devices()
    
    for dev in devices:
        loopback = " [green](LOOPBACK)[/green]" if dev['is_loopback'] else ""
        console.print(f"  [{dev['index']}] {dev['name']}{loopback}")
    
    best_device = recorder.find_best_device()
    if best_device is not None:
        console.print(f"\n[green]Auto-selected device: {best_device}[/green]")
    else:
        console.print("\n[yellow]No loopback device found. Using default microphone.[/yellow]")
        console.print("[dim]Tip: Enable 'Stereo Mix' in Windows Sound settings to capture system audio[/dim]")
    
    # Ask user to confirm or select device
    console.print("\n[cyan]Press Enter to start with selected device, or type device number:[/cyan]")
    try:
        user_input = input().strip()
        if user_input:
            best_device = int(user_input)
    except (ValueError, EOFError):
        pass
    
    # Start recording
    console.print("\n[bold green]Starting audio capture...[/bold green]")
    console.print("[dim]Make sure Google Meet audio is playing[/dim]\n")
    
    if not recorder.start(best_device):
        console.print("[red]Failed to start recording. Running transcript demo instead.[/red]")
        run_transcript_demo()
        return
    
    # Main recording loop
    console.print("[bold]Live Captions:[/bold]\n")
    
    try:
        while True:
            # Get audio chunk
            audio_chunk = recorder.get_chunk()
            
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Convert to WAV
                wav_bytes = recorder.audio_to_wav_bytes(audio_chunk)
                
                # Transcribe
                text = transcriber.transcribe_audio(wav_bytes)
                
                if text:
                    # Generate clean caption
                    caption = analyzer.generate_live_caption(text)
                    console.print(f"[cyan]>>> {caption}[/cyan]")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Stopping recording...[/yellow]")
    
    # Stop recording
    recorder.stop()
    
    # Get full transcript
    transcript = transcriber.get_full_transcript()
    
    if not transcript.strip():
        console.print("[yellow]No speech was detected. Make sure audio is playing.[/yellow]")
        return
    
    console.print(f"\n[green]✓ Captured {len(transcript)} characters of transcript[/green]")
    
    # Process with AI
    console.print("\n[cyan]Analyzing meeting with Gemini 2.5 Flash...[/cyan]")
    
    with console.status("[bold green]Generating meeting notes..."):
        caption = analyzer.generate_caption(transcript)
        summary = analyzer.generate_summary(transcript)
        action_items = analyzer.extract_action_items(transcript)
    
    # Display results
    display_results(caption, summary, action_items)
    
    # Save notes
    transcript_file, notes_file, json_file = save_meeting_notes(
        transcript, caption, summary, action_items
    )
    
    console.print(f"\n[green]✓ Meeting notes saved:[/green]")
    console.print(f"  • Transcript: {transcript_file}")
    console.print(f"  • Notes: {notes_file}")
    console.print(f"  • Data: {json_file}")


def run_transcript_demo():
    """Run demo with a sample transcript (no audio required)."""
    console.print("\n[cyan]Running with sample transcript...[/cyan]\n")
    
    sample_transcript = """
    John: Good morning everyone, let's get started with our product planning meeting.
    
    Sarah: Thanks John. So we need to discuss the Q1 roadmap and finalize the feature priorities.
    
    John: Right. I think we should focus on the mobile app improvements first. User feedback has been clear about that.
    
    Mike: I agree. The performance issues on Android are critical. We're seeing a 20% drop-off rate.
    
    Sarah: Okay, so mobile performance is priority one. What about the new dashboard feature?
    
    John: Let's push that to Q2. We don't have the bandwidth right now.
    
    Mike: Makes sense. I can have the Android fixes ready by end of January if we start next week.
    
    Sarah: Perfect. Mike, you'll lead the mobile performance sprint. John, can you handle the user research for the dashboard?
    
    John: Yes, I'll set up interviews with 10 power users this month.
    
    Sarah: Great. Let's also not forget about the API documentation. Developers have been complaining.
    
    Mike: I can assign that to the junior devs as a side project.
    
    John: Good idea. So to summarize - mobile performance first, dashboard research ongoing, and API docs as a side task.
    
    Sarah: Exactly. Let's reconvene in two weeks to check progress. Meeting adjourned.
    """
    
    analyzer = MeetingAnalyzer()
    
    with console.status("[bold green]Analyzing meeting with Gemini 2.5 Flash..."):
        caption = analyzer.generate_caption(sample_transcript)
        summary = analyzer.generate_summary(sample_transcript)
        action_items = analyzer.extract_action_items(sample_transcript)
    
    # Display results
    display_results(caption, summary, action_items)
    
    # Save notes
    transcript_file, notes_file, json_file = save_meeting_notes(
        sample_transcript, caption, summary, action_items
    )
    
    console.print(f"\n[green]✓ Meeting notes saved:[/green]")
    console.print(f"  • Transcript: {transcript_file}")
    console.print(f"  • Notes: {notes_file}")
    console.print(f"  • Data: {json_file}")


if __name__ == "__main__":
    run_demo()
