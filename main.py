"""
MinutesX - Live Google Meet AI Notes (Multi-Agent System)
==========================================================

LIVE MODE for real Google Meet transcription and AI-powered meeting notes.
Uses multi-agent architecture with A2A protocol.

Commands:
    python main.py live      # Start live Google Meet capture
    python main.py process   # Process existing transcript file
    python main.py devices   # List audio devices
    python main.py config    # Show configuration

Output: ./output/ai_caption.txt, ai_summary.txt, ai_notes.txt
"""
import os
import sys
import time
import json
import queue
import base64
import io
import wave
import uuid
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

# Load environment
from dotenv import load_dotenv
load_dotenv()

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
OUTPUT_DIR = Path("./output")

# =============================================================================
# Configuration Check
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    console.print(Panel.fit(
        "[red]❌ GOOGLE_API_KEY not found![/red]\n\n"
        "[yellow]Add to your .env file:[/yellow]\n"
        "GOOGLE_API_KEY=your_api_key_here\n\n"
        "[cyan]Get free key: https://aistudio.google.com/apikey[/cyan]",
        title="Configuration Error",
        border_style="red"
    ))
    sys.exit(1)

# Initialize Gemini
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash-exp"

# Audio libraries
AUDIO_OK = False
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_OK = True
except ImportError:
    pass


# =============================================================================
# A2A Protocol - Agent Communication
# =============================================================================

class MessageType(Enum):
    TASK_REQUEST = "TASK_REQUEST"
    TASK_RESULT = "TASK_RESULT"
    STATUS_UPDATE = "STATUS_UPDATE"


@dataclass
class A2AMessage:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.TASK_REQUEST
    source_agent: str = ""
    target_agent: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class A2AMessageBus:
    def __init__(self):
        self.subscribers: Dict[str, Callable] = {}
        self.message_log: List[A2AMessage] = []
        self._lock = threading.Lock()
    
    def subscribe(self, agent_id: str, handler: Callable):
        with self._lock:
            self.subscribers[agent_id] = handler
    
    def publish(self, message: A2AMessage) -> bool:
        with self._lock:
            self.message_log.append(message)
        if message.target_agent in self.subscribers:
            try:
                self.subscribers[message.target_agent](message)
                return True
            except:
                pass
        return False
    
    def send_task(self, source: str, target: str, task: str, data: Any) -> A2AMessage:
        msg = A2AMessage(
            message_type=MessageType.TASK_REQUEST,
            source_agent=source, target_agent=target,
            payload={"task": task, "data": data}
        )
        self.publish(msg)
        return msg
    
    def get_stats(self) -> Dict:
        return {"total_messages": len(self.message_log), "agents": list(self.subscribers.keys())}


message_bus = A2AMessageBus()


# =============================================================================
# Multi-Agent System
# =============================================================================

class BaseAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.model = genai.GenerativeModel(MODEL_NAME)
        message_bus.subscribe(agent_id, self._handle_message)
    
    def _handle_message(self, message: A2AMessage):
        task = message.payload.get("task")
        data = message.payload.get("data")
        if hasattr(self, f"task_{task}"):
            return getattr(self, f"task_{task}")(data)
    
    def generate(self, prompt: str) -> str:
        try:
            return self.model.generate_content(prompt).text.strip()
        except Exception as e:
            return f"Error: {e}"


class CaptionAgent(BaseAgent):
    PROMPT = """Create professional captions for this meeting. Return JSON:
{{"headline": "max 80 chars", "one_liner": "max 140 chars", "slack_update": "1-2 sentences", "email_subject": "max 60 chars"}}
TRANSCRIPT: {transcript}
Return ONLY valid JSON."""

    def __init__(self):
        super().__init__("caption_agent")
    
    def generate_caption(self, transcript: str) -> Dict:
        response = self.generate(self.PROMPT.format(transcript=transcript[:8000]))
        try:
            text = response.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except:
            return {"headline": "Meeting Summary", "one_liner": "Meeting completed", 
                   "slack_update": "Meeting notes ready!", "email_subject": "Meeting Notes"}


class SummaryAgent(BaseAgent):
    PROMPT = """Analyze meeting and create summary. Return JSON:
{{"executive_summary": "3-4 sentences", "key_points": ["point1", "point2", "point3", "point4", "point5"],
"decisions": ["decision1"], "topics": ["topic1", "topic2"], "participants": ["person1"], "outcome": "brief outcome"}}
TRANSCRIPT: {transcript}
Return ONLY valid JSON."""

    def __init__(self):
        super().__init__("summary_agent")
    
    def generate_summary(self, transcript: str) -> Dict:
        response = self.generate(self.PROMPT.format(transcript=transcript[:15000]))
        try:
            text = response.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except:
            return {"executive_summary": "Summary unavailable", "key_points": [], 
                   "decisions": [], "topics": [], "participants": [], "outcome": "See transcript"}


class ActionAgent(BaseAgent):
    PROMPT = """Extract action items from meeting. Return JSON:
{{"action_items": [{{"task": "description", "owner": "name", "priority": "high|medium|low", "due": "date or TBD"}}],
"follow_ups": ["item1"], "next_steps": ["step1"]}}
TRANSCRIPT: {transcript}
Return ONLY valid JSON."""

    def __init__(self):
        super().__init__("action_agent")
    
    def extract_actions(self, transcript: str) -> Dict:
        response = self.generate(self.PROMPT.format(transcript=transcript[:15000]))
        try:
            text = response.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except:
            return {"action_items": [], "follow_ups": [], "next_steps": []}


class TranscriptionAgent(BaseAgent):
    def __init__(self):
        super().__init__("transcription_agent")
    
    def transcribe(self, wav_bytes: bytes) -> Optional[str]:
        try:
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
            response = self.model.generate_content([
                "Transcribe accurately. Return ONLY spoken words. If silence, return [silence].",
                {"mime_type": "audio/wav", "data": audio_b64}
            ])
            text = response.text.strip()
            return text if text and text != "[silence]" and len(text) > 3 else None
        except:
            return None


class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("orchestrator")
        self.caption_agent = CaptionAgent()
        self.summary_agent = SummaryAgent()
        self.action_agent = ActionAgent()
        self.transcription_agent = TranscriptionAgent()
    
    def process_meeting(self, transcript: str, meeting_id: str = None) -> Dict:
        meeting_id = meeting_id or str(uuid.uuid4())[:8]
        start = time.time()
        
        # Send A2A messages
        message_bus.send_task("orchestrator", "caption_agent", "generate", {"transcript": transcript})
        message_bus.send_task("orchestrator", "summary_agent", "generate", {"transcript": transcript})
        message_bus.send_task("orchestrator", "action_agent", "generate", {"transcript": transcript})
        
        # Parallel execution
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.caption_agent.generate_caption, transcript): "caption",
                executor.submit(self.summary_agent.generate_summary, transcript): "summary",
                executor.submit(self.action_agent.extract_actions, transcript): "action",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except:
                    results[name] = {}
        
        return {
            "meeting_id": meeting_id,
            "processed_at": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - start, 2),
            "transcript_length": len(transcript),
            "caption": results.get("caption", {}),
            "summary": results.get("summary", {}),
            "actions": results.get("action", {}),
            "a2a_stats": message_bus.get_stats(),
        }


# =============================================================================
# Audio Capture
# =============================================================================

class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_seconds=8):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.queue = queue.Queue()
        self.recording = False
        self.duration = 0
    
    def list_devices(self):
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                name = d['name'].lower()
                is_loopback = any(x in name for x in ['loopback', 'stereo mix', 'wasapi', 'mix'])
                devices.append({'idx': i, 'name': d['name'], 'loopback': is_loopback})
        return devices
    
    def find_loopback(self):
        for d in self.list_devices():
            if d['loopback']:
                return d['idx']
        return None
    
    def start(self, device=None):
        self.recording = True
        self.duration = 0
        def callback(indata, frames, time_info, status):
            if self.recording:
                self.queue.put(indata.copy())
        try:
            self.stream = sd.InputStream(device=device, channels=1, samplerate=self.sample_rate,
                                        dtype='int16', callback=callback, blocksize=int(self.sample_rate * 0.5))
            self.stream.start()
            return True
        except:
            return False
    
    def get_chunk(self):
        chunks = []
        samples_needed = int(self.sample_rate * self.chunk_seconds)
        collected = 0
        while collected < samples_needed and self.recording:
            try:
                chunk = self.queue.get(timeout=0.5)
                chunks.append(chunk)
                collected += len(chunk)
            except:
                break
        if chunks:
            audio = np.concatenate(chunks)
            self.duration += len(audio) / self.sample_rate
            return audio
        return None
    
    def stop(self):
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def to_wav(self, audio):
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(audio.tobytes())
        return buf.getvalue()
    
    def get_duration_str(self):
        mins, secs = divmod(int(self.duration), 60)
        return f"{mins:02d}:{secs:02d}"


# =============================================================================
# Output Functions
# =============================================================================

def save_outputs(result: Dict, transcript: str) -> Dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    caption_data = result.get("caption", {})
    summary_data = result.get("summary", {})
    action_data = result.get("actions", {})
    
    # AI Caption
    caption_file = OUTPUT_DIR / "ai_caption.txt"
    caption_file.write_text(f"""MinutesX - AI CAPTION
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Meeting ID: {result.get('meeting_id')}

HEADLINE: {caption_data.get('headline', 'Meeting Summary')}
ONE-LINER: {caption_data.get('one_liner', '')}
SLACK: {caption_data.get('slack_update', '')}
EMAIL: {caption_data.get('email_subject', '')}

Multi-Agent System | A2A Protocol
""", encoding='utf-8')
    
    # AI Summary
    summary_file = OUTPUT_DIR / "ai_summary.txt"
    key_points = "\n".join([f"  • {p}" for p in summary_data.get('key_points', [])])
    decisions = "\n".join([f"  ✓ {d}" for d in summary_data.get('decisions', [])]) or "  None"
    
    summary_file.write_text(f"""MinutesX - AI SUMMARY
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

EXECUTIVE SUMMARY:
{summary_data.get('executive_summary', 'N/A')}

KEY POINTS:
{key_points or '  None'}

DECISIONS:
{decisions}

TOPICS: {', '.join(summary_data.get('topics', []))}
PARTICIPANTS: {', '.join(summary_data.get('participants', []))}
OUTCOME: {summary_data.get('outcome', '')}

Multi-Agent System | A2A Protocol
""", encoding='utf-8')
    
    # AI Notes
    notes_file = OUTPUT_DIR / "ai_notes.txt"
    actions_str = ""
    for i, item in enumerate(action_data.get('action_items', []), 1):
        p = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.get('priority', 'medium'), "⚪")
        actions_str += f"\n  {p} [{i}] {item.get('task', '')}\n      Owner: {item.get('owner', 'TBD')} | Due: {item.get('due', 'TBD')}"
    
    follow_ups = "\n".join([f"  • {f}" for f in action_data.get('follow_ups', [])]) or "  None"
    next_steps = "\n".join([f"  {i}. {s}" for i, s in enumerate(action_data.get('next_steps', []), 1)]) or "  None"
    
    notes_file.write_text(f"""MinutesX - AI MEETING NOTES
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Meeting ID: {result.get('meeting_id')}

ACTION ITEMS:
{actions_str or '  None identified'}

FOLLOW-UP REQUIRED:
{follow_ups}

NEXT STEPS:
{next_steps}

Processing: {result.get('duration_seconds', 0)}s
A2A Messages: {result.get('a2a_stats', {}).get('total_messages', 0)}

Multi-Agent System | Orchestrator + Caption + Summary + Action Agents
""", encoding='utf-8')
    
    # Transcript
    transcript_file = OUTPUT_DIR / f"transcript_{timestamp}.txt"
    transcript_file.write_text(f"Meeting Transcript - {datetime.now()}\n\n{transcript}", encoding='utf-8')
    
    return {'caption': caption_file, 'summary': summary_file, 'notes': notes_file, 'transcript': transcript_file}


# =============================================================================
# CLI Commands
# =============================================================================

@click.group()
@click.version_option(version="1.0.0", prog_name="MinutesX")
def cli():
    """MinutesX - Live Google Meet AI Notes (Multi-Agent System)"""
    pass


@cli.command()
@click.option("--device", "-d", default=None, type=int, help="Audio device index")
def live(device: Optional[int]):
    """
    🎤 LIVE MODE - Capture Google Meet audio in real-time.
    
    Listens to system audio, transcribes with Gemini, and generates
    AI-powered meeting notes using multi-agent system.
    
    Press Ctrl+C to stop and generate notes.
    """
    if not AUDIO_OK:
        console.print("[red]❌ Audio libraries not installed![/red]")
        console.print("[yellow]Run: pip install sounddevice numpy[/yellow]")
        return
    
    console.print(Panel.fit(
        "[bold cyan]🎤 LIVE MODE - Google Meet + Multi-Agent AI[/bold cyan]\n\n"
        "• Captures system audio from Google Meet\n"
        "• Transcribes with Gemini 2.5 Flash\n"
        "• Processes through 4-agent pipeline\n\n"
        "[yellow]Press Ctrl+C to stop and generate notes[/yellow]",
        title="MinutesX Live",
        border_style="cyan"
    ))
    
    orchestrator = OrchestratorAgent()
    capture = AudioCapture(chunk_seconds=8)
    
    # Show devices
    console.print("\n[bold cyan]📢 Audio Devices:[/bold cyan]")
    devices = capture.list_devices()
    
    table = Table(show_header=True)
    table.add_column("#", width=4)
    table.add_column("Device", width=50)
    table.add_column("Type", width=12)
    
    for d in devices:
        dtype = "[green]LOOPBACK[/green]" if d['loopback'] else "[dim]Mic[/dim]"
        table.add_row(str(d['idx']), d['name'], dtype)
    console.print(table)
    
    # Select device
    loopback = capture.find_loopback()
    if device is None:
        if loopback:
            console.print(f"\n[green]✓ Auto-detected loopback: [{loopback}][/green]")
        console.print("\n[cyan]Enter device # (or Enter for auto):[/cyan] ", end="")
        try:
            inp = input().strip()
            device = int(inp) if inp else loopback
        except:
            device = loopback
    
    if not capture.start(device):
        console.print("[red]Failed to start audio capture.[/red]")
        return
    
    console.print("\n" + "="*60)
    console.print("[bold red]🔴 RECORDING[/bold red] - Listening to Google Meet...")
    console.print("[yellow]Press Ctrl+C when meeting ends[/yellow]")
    console.print("="*60 + "\n")
    
    transcript_parts = []
    
    try:
        while True:
            audio = capture.get_chunk()
            if audio is not None and len(audio) > 0:
                wav = capture.to_wav(audio)
                console.print(f"[dim]⏱ {capture.get_duration_str()}[/dim] ", end="")
                
                text = orchestrator.transcription_agent.transcribe(wav)
                if text:
                    transcript_parts.append(text)
                    console.print(f"[cyan]{text[:100]}{'...' if len(text) > 100 else ''}[/cyan]")
                else:
                    console.print("[dim](listening...)[/dim]")
            time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⏹ Stopping...[/yellow]")
    
    capture.stop()
    transcript = "\n".join(transcript_parts)
    
    if not transcript.strip():
        console.print("[yellow]No speech detected.[/yellow]")
        return
    
    console.print(f"\n[green]✓ Captured {len(transcript)} characters[/green]")
    console.print("\n[bold]🤖 Processing with multi-agent system...[/bold]")
    
    result = orchestrator.process_meeting(transcript)
    files = save_outputs(result, transcript)
    
    # Display results
    console.print(Panel(f"[cyan]{result['caption'].get('headline', 'Meeting')}[/cyan]", title="📝 Caption"))
    console.print(Panel(f"{result['summary'].get('executive_summary', 'N/A')[:500]}", title="📋 Summary"))
    
    console.print(Panel.fit(
        f"[green]✅ Files saved to {OUTPUT_DIR.absolute()}[/green]\n\n"
        f"📝 ai_caption.txt\n📋 ai_summary.txt\n📄 ai_notes.txt",
        title="Output", border_style="green"
    ))


@cli.command()
@click.argument("transcript_file", type=click.Path(exists=True))
def process(transcript_file: str):
    """
    📄 Process an existing transcript file.
    
    Example: python main.py process ./transcript.txt
    """
    console.print(f"[cyan]Processing: {transcript_file}[/cyan]")
    
    transcript = Path(transcript_file).read_text(encoding='utf-8')
    if not transcript.strip():
        console.print("[red]Empty transcript file.[/red]")
        return
    
    orchestrator = OrchestratorAgent()
    console.print("[bold]🤖 Processing with multi-agent system...[/bold]")
    
    result = orchestrator.process_meeting(transcript)
    files = save_outputs(result, transcript)
    
    console.print(Panel.fit(f"[green]✅ Files saved to {OUTPUT_DIR}[/green]", border_style="green"))


@cli.command()
def devices():
    """📢 List audio devices."""
    if not AUDIO_OK:
        console.print("[red]Audio libraries not installed. Run: pip install sounddevice numpy[/red]")
        return
    
    capture = AudioCapture()
    devices = capture.list_devices()
    
    table = Table(title="Audio Devices", show_header=True)
    table.add_column("#", width=4)
    table.add_column("Device Name", width=55)
    table.add_column("Type", width=12)
    
    for d in devices:
        dtype = "[green]LOOPBACK[/green]" if d['loopback'] else "[dim]Microphone[/dim]"
        table.add_row(str(d['idx']), d['name'], dtype)
    
    console.print(table)
    
    loopback = capture.find_loopback()
    if loopback:
        console.print(f"\n[green]✓ Recommended for Google Meet: Device #{loopback}[/green]")
    else:
        console.print("\n[yellow]⚠ No loopback device found. Enable 'Stereo Mix' in Windows Sound settings.[/yellow]")


@cli.command()
def config():
    """⚙️ Show configuration."""
    table = Table(title="MinutesX Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    table.add_row("API Key", "✓ Set" if GOOGLE_API_KEY else "✗ Missing")
    table.add_row("Model", MODEL_NAME)
    table.add_row("Audio", "✓ Available" if AUDIO_OK else "✗ Not installed")
    table.add_row("Output Dir", str(OUTPUT_DIR.absolute()))
    
    console.print(table)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
