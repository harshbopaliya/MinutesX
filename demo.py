"""
MinutesX - Live Google Meet AI Notes (Multi-Agent System)
==========================================================

ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────────┐
  │                    ORCHESTRATOR AGENT                       │
  │                  (Coordinates all agents)                   │
  └─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │  CAPTION   │    │  SUMMARY   │    │   ACTION   │
    │   AGENT    │    │   AGENT    │    │   AGENT    │
    └────────────┘    └────────────┘    └────────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                              ▼
                    ┌────────────────┐
                    │ A2A MESSAGE BUS │
                    │  (Coordination) │
                    └────────────────┘

OUTPUT FILES:
  ./output/ai_caption.txt  - Meeting headline & captions
  ./output/ai_summary.txt  - Executive summary & key points
  ./output/ai_notes.txt    - Detailed notes & action items

POWERED BY:
  - Google Gemini 2.5 Flash (gemini-2.0-flash-exp)
  - Google ADK Multi-Agent Pattern
  - A2A (Agent-to-Agent) Protocol

Run: python demo.py
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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# =============================================================================
# Configuration
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OUTPUT_DIR = Path("./output")

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

# Audio libraries (optional)
AUDIO_OK = False
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_OK = True
except ImportError:
    pass


# =============================================================================
# A2A Protocol - Agent-to-Agent Communication
# =============================================================================

class MessageType(Enum):
    """Types of A2A messages."""
    TASK_REQUEST = "TASK_REQUEST"
    TASK_RESULT = "TASK_RESULT"
    STATUS_UPDATE = "STATUS_UPDATE"
    ERROR = "ERROR"


@dataclass
class A2AMessage:
    """A2A Protocol message for agent communication."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.TASK_REQUEST
    source_agent: str = ""
    target_agent: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.message_id,
            "type": self.message_type.value,
            "from": self.source_agent,
            "to": self.target_agent,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


class A2AMessageBus:
    """Message bus for A2A agent communication."""
    
    def __init__(self):
        self.subscribers: Dict[str, Callable] = {}
        self.message_log: List[A2AMessage] = []
        self._lock = threading.Lock()
    
    def subscribe(self, agent_id: str, handler: Callable):
        """Subscribe agent to receive messages."""
        with self._lock:
            self.subscribers[agent_id] = handler
    
    def publish(self, message: A2AMessage) -> bool:
        """Publish message to target agent."""
        with self._lock:
            self.message_log.append(message)
        
        target = message.target_agent
        if target in self.subscribers:
            try:
                self.subscribers[target](message)
                return True
            except Exception as e:
                console.print(f"[dim]A2A Error: {e}[/dim]")
        return False
    
    def send_task(self, source: str, target: str, task: str, data: Any) -> A2AMessage:
        """Send a task request to an agent."""
        msg = A2AMessage(
            message_type=MessageType.TASK_REQUEST,
            source_agent=source,
            target_agent=target,
            payload={"task": task, "data": data}
        )
        self.publish(msg)
        return msg
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_messages": len(self.message_log),
            "agents": list(self.subscribers.keys()),
        }


# Global message bus
message_bus = A2AMessageBus()


# =============================================================================
# Base Agent Class (Google ADK Pattern)
# =============================================================================

class BaseAgent:
    """Base class for all MinutesX agents following Google ADK pattern."""
    
    def __init__(self, agent_id: str, model_name: str = MODEL_NAME):
        self.agent_id = agent_id
        self.model = genai.GenerativeModel(model_name)
        self.results: Dict[str, Any] = {}
        
        # Register with message bus
        message_bus.subscribe(agent_id, self._handle_message)
    
    def _handle_message(self, message: A2AMessage):
        """Handle incoming A2A message."""
        task = message.payload.get("task")
        data = message.payload.get("data")
        
        if hasattr(self, f"task_{task}"):
            result = getattr(self, f"task_{task}")(data)
            # Send result back
            response = A2AMessage(
                message_type=MessageType.TASK_RESULT,
                source_agent=self.agent_id,
                target_agent=message.source_agent,
                payload={"task": task, "result": result}
            )
            message_bus.publish(response)
    
    def generate(self, prompt: str) -> str:
        """Generate response using Gemini."""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"


# =============================================================================
# Specialized Agents
# =============================================================================

class CaptionAgent(BaseAgent):
    """Agent for generating meeting captions and headlines."""
    
    PROMPT = """You are the Caption Agent for MinutesX meeting notes system.
    
Create professional captions for this meeting transcript.

TRANSCRIPT:
{transcript}

Generate captions in this EXACT JSON format:
{{
    "headline": "Punchy headline (max 80 chars)",
    "one_liner": "One-line summary (max 140 chars)",
    "slack_update": "Casual Slack message (1-2 sentences)",
    "email_subject": "Email subject line (max 60 chars)"
}}

Return ONLY valid JSON."""

    def __init__(self):
        super().__init__("caption_agent")
    
    def generate_caption(self, transcript: str) -> Dict[str, Any]:
        """Generate meeting captions."""
        prompt = self.PROMPT.format(transcript=transcript[:8000])
        response = self.generate(prompt)
        
        try:
            # Clean and parse JSON
            text = response.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except:
            return {
                "headline": f"Meeting - {datetime.now().strftime('%B %d')}",
                "one_liner": "Team meeting completed successfully",
                "slack_update": "Meeting notes are ready for review!",
                "email_subject": "Meeting Notes Available"
            }
    
    def task_generate(self, data: Dict) -> Dict[str, Any]:
        """A2A task handler."""
        return self.generate_caption(data.get("transcript", ""))


class SummaryAgent(BaseAgent):
    """Agent for generating meeting summaries."""
    
    PROMPT = """You are the Summary Agent for MinutesX meeting notes system.

Analyze this meeting transcript and create a comprehensive summary.

TRANSCRIPT:
{transcript}

Generate summary in this EXACT JSON format:
{{
    "executive_summary": "3-4 sentence executive summary",
    "key_points": [
        "Key point 1 with context",
        "Key point 2 with context",
        "Key point 3 with context",
        "Key point 4 with context",
        "Key point 5 with context"
    ],
    "decisions": [
        "Decision 1 that was made",
        "Decision 2 that was made"
    ],
    "topics": ["Topic 1", "Topic 2", "Topic 3"],
    "participants": ["Person 1", "Person 2"],
    "outcome": "Brief meeting outcome"
}}

Return ONLY valid JSON."""

    def __init__(self):
        super().__init__("summary_agent")
    
    def generate_summary(self, transcript: str) -> Dict[str, Any]:
        """Generate meeting summary."""
        prompt = self.PROMPT.format(transcript=transcript[:15000])
        response = self.generate(prompt)
        
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except:
            return {
                "executive_summary": "Meeting summary could not be generated.",
                "key_points": ["Please review transcript"],
                "decisions": [],
                "topics": [],
                "participants": [],
                "outcome": "See transcript for details"
            }
    
    def task_generate(self, data: Dict) -> Dict[str, Any]:
        return self.generate_summary(data.get("transcript", ""))


class ActionAgent(BaseAgent):
    """Agent for extracting action items and tasks."""
    
    PROMPT = """You are the Action Item Agent for MinutesX meeting notes system.

Extract ALL action items, tasks, and commitments from this meeting.

TRANSCRIPT:
{transcript}

Generate action items in this EXACT JSON format:
{{
    "action_items": [
        {{
            "task": "Clear task description",
            "owner": "Person name or Unassigned",
            "priority": "high|medium|low",
            "due": "Date if mentioned or TBD",
            "context": "Brief context"
        }}
    ],
    "follow_ups": [
        "Follow-up item 1",
        "Follow-up item 2"
    ],
    "next_steps": [
        "Next step 1",
        "Next step 2"
    ]
}}

Look for: "I will...", "Please...", "Can you...", "We need to...", "Action item:", deadlines, commitments.

Return ONLY valid JSON."""

    def __init__(self):
        super().__init__("action_agent")
    
    def extract_actions(self, transcript: str) -> Dict[str, Any]:
        """Extract action items from transcript."""
        prompt = self.PROMPT.format(transcript=transcript[:15000])
        response = self.generate(prompt)
        
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except:
            return {
                "action_items": [],
                "follow_ups": ["Review transcript for action items"],
                "next_steps": []
            }
    
    def task_generate(self, data: Dict) -> Dict[str, Any]:
        return self.extract_actions(data.get("transcript", ""))


class TranscriptionAgent(BaseAgent):
    """Agent for transcribing audio using Gemini."""
    
    def __init__(self):
        super().__init__("transcription_agent")
    
    def transcribe(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribe audio to text."""
        try:
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
            response = self.model.generate_content([
                """Transcribe this audio accurately.
                - Return ONLY spoken words
                - Include speaker names if identifiable
                - If silence, return [silence]""",
                {"mime_type": "audio/wav", "data": audio_b64}
            ])
            text = response.text.strip()
            if text and text != "[silence]" and len(text) > 3:
                return text
            return None
        except:
            return None


# =============================================================================
# Orchestrator Agent - Coordinates All Agents
# =============================================================================

class OrchestratorAgent(BaseAgent):
    """
    Main orchestrator agent that coordinates all sub-agents.
    
    Implements Google ADK multi-agent pattern:
    - Parallel execution of independent agents
    - A2A protocol for agent communication
    - Result aggregation and formatting
    """
    
    def __init__(self):
        super().__init__("orchestrator")
        
        # Initialize sub-agents
        self.caption_agent = CaptionAgent()
        self.summary_agent = SummaryAgent()
        self.action_agent = ActionAgent()
        self.transcription_agent = TranscriptionAgent()
        
        console.print("[green]✓[/green] Orchestrator initialized with 4 agents")
    
    def process_meeting(self, transcript: str, meeting_id: str = None) -> Dict[str, Any]:
        """
        Process meeting through multi-agent pipeline.
        
        Pipeline:
        1. Parallel: Caption + Summary + Action agents
        2. Aggregate results
        3. Format outputs
        """
        meeting_id = meeting_id or str(uuid.uuid4())[:8]
        start_time = time.time()
        
        console.print(f"\n[cyan]🤖 Orchestrator processing meeting {meeting_id}[/cyan]")
        console.print(f"[dim]Transcript: {len(transcript)} characters[/dim]\n")
        
        # Send A2A messages to agents (for demonstration)
        message_bus.send_task("orchestrator", "caption_agent", "generate", {"transcript": transcript})
        message_bus.send_task("orchestrator", "summary_agent", "generate", {"transcript": transcript})
        message_bus.send_task("orchestrator", "action_agent", "generate", {"transcript": transcript})
        
        # Execute agents in parallel
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Define parallel tasks
            tasks = {
                "caption": (self.caption_agent.generate_caption, "Generating captions..."),
                "summary": (self.summary_agent.generate_summary, "Generating summary..."),
                "action": (self.action_agent.extract_actions, "Extracting actions..."),
            }
            
            # Run in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                for name, (func, desc) in tasks.items():
                    task_id = progress.add_task(desc, total=None)
                    futures[executor.submit(func, transcript)] = (name, task_id)
                
                for future in as_completed(futures):
                    name, task_id = futures[future]
                    try:
                        results[name] = future.result()
                        progress.update(task_id, description=f"[green]✓[/green] {name.title()} complete")
                    except Exception as e:
                        results[name] = {"error": str(e)}
                        progress.update(task_id, description=f"[red]✗[/red] {name.title()} failed")
        
        # Calculate processing time
        duration = time.time() - start_time
        
        # Aggregate results
        final_result = {
            "meeting_id": meeting_id,
            "processed_at": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "transcript_length": len(transcript),
            "caption": results.get("caption", {}),
            "summary": results.get("summary", {}),
            "actions": results.get("action", {}),
            "a2a_stats": message_bus.get_stats(),
        }
        
        console.print(f"\n[green]✓ Processing complete in {duration:.1f}s[/green]")
        console.print(f"[dim]A2A Messages: {message_bus.get_stats()['total_messages']}[/dim]")
        
        return final_result


# =============================================================================
# Audio Capture (for Live Mode)
# =============================================================================

class AudioCapture:
    """Captures system audio for Google Meet."""
    
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
            self.stream = sd.InputStream(
                device=device, channels=1, samplerate=self.sample_rate,
                dtype='int16', callback=callback, blocksize=int(self.sample_rate * 0.5)
            )
            self.stream.start()
            return True
        except Exception as e:
            console.print(f"[red]Audio error: {e}[/red]")
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
# Audio Capture Class - Enhanced for Google Meet
# =============================================================================

class AudioCapture:
    """Captures system audio from Google Meet using loopback/WASAPI."""
    
    def __init__(self, sample_rate=16000, chunk_seconds=8):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds  # Shorter chunks for faster response
        self.queue = queue.Queue()
        self.recording = False
        self.all_audio = []
        self.total_duration = 0
    
    def list_devices(self):
        """List all audio input devices with loopback detection."""
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                name_lower = d['name'].lower()
                # Detect loopback devices (captures system audio)
                is_loopback = any(x in name_lower for x in [
                    'loopback', 'stereo mix', 'wasapi', 'what u hear',
                    'wave out', 'mix', 'system audio', 'virtual'
                ])
                devices.append({
                    'idx': i, 
                    'name': d['name'], 
                    'loopback': is_loopback,
                    'channels': d['max_input_channels'],
                    'rate': d['default_samplerate']
                })
        return devices
    
    def find_loopback(self):
        """Find the best loopback device for capturing system audio."""
        devices = self.list_devices()
        
        # Priority order for loopback devices
        priority_keywords = ['loopback', 'stereo mix', 'wasapi', 'what u hear']
        
        for keyword in priority_keywords:
            for d in devices:
                if keyword in d['name'].lower():
                    return d['idx']
        
        # If no loopback found, return None (will use mic)
        return None
    
    def start(self, device=None):
        """Start audio capture from selected device."""
        self.recording = True
        self.all_audio = []
        self.total_duration = 0
        
        def callback(indata, frames, time_info, status):
            if self.recording:
                self.queue.put(indata.copy())
        
        try:
            # Use specified device or default
            self.stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=self.sample_rate,
                dtype='int16',
                callback=callback,
                blocksize=int(self.sample_rate * 0.5)  # 500ms blocks
            )
            self.stream.start()
            return True
        except Exception as e:
            console.print(f"[red]Audio error: {e}[/red]")
            return False
    
    def get_chunk(self):
        """Get audio chunk for transcription."""
        chunks = []
        samples_needed = int(self.sample_rate * self.chunk_seconds)
        collected = 0
        
        timeout_start = time.time()
        while collected < samples_needed and self.recording:
            if time.time() - timeout_start > self.chunk_seconds + 2:
                break
            try:
                chunk = self.queue.get(timeout=0.5)
                chunks.append(chunk)
                collected += len(chunk)
            except queue.Empty:
                continue
        
        if chunks:
            audio = np.concatenate(chunks)
            self.all_audio.append(audio)
            self.total_duration += len(audio) / self.sample_rate
            return audio
        return None
    
    def stop(self):
        """Stop audio capture."""
        self.recording = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
    
    def get_duration_str(self):
        """Get formatted duration string."""
        mins = int(self.total_duration // 60)
        secs = int(self.total_duration % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def to_wav(self, audio):
        """Convert numpy audio to WAV bytes for Gemini."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(audio.tobytes())
        return buf.getvalue()


# =============================================================================
# Output Generation - Save to ./output/ folder
# =============================================================================

def save_outputs(result: Dict[str, Any], transcript: str):
    """Save all outputs to ./output folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    caption_data = result.get("caption", {})
    summary_data = result.get("summary", {})
    action_data = result.get("actions", {})
    
    # 1. AI CAPTION
    caption_file = OUTPUT_DIR / "ai_caption.txt"
    caption_content = f"""╔══════════════════════════════════════════════════════════════════════╗
║              MinutesX - AI CAPTION (Multi-Agent System)              ║
╚══════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Meeting ID: {result.get('meeting_id', 'N/A')}
Model: Gemini 2.5 Flash (Multi-Agent)
Processing Time: {result.get('duration_seconds', 0)}s

═══════════════════════════════════════════════════════════════════════

📰 HEADLINE:
{caption_data.get('headline', 'Meeting Summary')}

📝 ONE-LINER:
{caption_data.get('one_liner', 'Meeting completed')}

💬 SLACK UPDATE:
{caption_data.get('slack_update', 'Meeting notes ready!')}

📧 EMAIL SUBJECT:
{caption_data.get('email_subject', 'Meeting Notes')}

═══════════════════════════════════════════════════════════════════════

AGENT: CaptionAgent
A2A PROTOCOL: Task completed via message bus

Powered by MinutesX | github.com/harshbopaliya/MinutesX
"""
    caption_file.write_text(caption_content, encoding='utf-8')
    
    # 2. AI SUMMARY
    summary_file = OUTPUT_DIR / "ai_summary.txt"
    
    key_points_str = "\n".join([f"  • {p}" for p in summary_data.get('key_points', [])])
    decisions_str = "\n".join([f"  ✓ {d}" for d in summary_data.get('decisions', [])]) or "  (No decisions recorded)"
    topics_str = ", ".join(summary_data.get('topics', [])) or "General discussion"
    participants_str = ", ".join(summary_data.get('participants', [])) or "Not identified"
    
    summary_content = f"""╔══════════════════════════════════════════════════════════════════════╗
║              MinutesX - AI SUMMARY (Multi-Agent System)              ║
╚══════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Meeting ID: {result.get('meeting_id', 'N/A')}
Model: Gemini 2.5 Flash (Multi-Agent)

═══════════════════════════════════════════════════════════════════════

## EXECUTIVE SUMMARY

{summary_data.get('executive_summary', 'Summary not available.')}

═══════════════════════════════════════════════════════════════════════

## KEY DISCUSSION POINTS

{key_points_str or '  • No key points extracted'}

═══════════════════════════════════════════════════════════════════════

## DECISIONS MADE

{decisions_str}

═══════════════════════════════════════════════════════════════════════

## TOPICS COVERED

{topics_str}

## PARTICIPANTS

{participants_str}

## MEETING OUTCOME

{summary_data.get('outcome', 'See transcript for details.')}

═══════════════════════════════════════════════════════════════════════

AGENT: SummaryAgent
A2A PROTOCOL: Task completed via message bus

Powered by MinutesX | github.com/harshbopaliya/MinutesX
"""
    summary_file.write_text(summary_content, encoding='utf-8')
    
    # 3. AI NOTES (with Action Items)
    notes_file = OUTPUT_DIR / "ai_notes.txt"
    
    action_items = action_data.get('action_items', [])
    actions_str = ""
    for i, item in enumerate(action_items, 1):
        priority_mark = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.get('priority', 'medium'), "⚪")
        actions_str += f"""
  {priority_mark} [{i}] {item.get('task', 'Task')}
      → Assigned to: {item.get('owner', 'Unassigned')}
      → Priority: {item.get('priority', 'medium').upper()}
      → Due: {item.get('due', 'TBD')}
      → Context: {item.get('context', '')}
"""
    
    if not actions_str:
        actions_str = "  No specific action items identified.\n"
    
    follow_ups = action_data.get('follow_ups', [])
    follow_ups_str = "\n".join([f"  • {f}" for f in follow_ups]) or "  None"
    
    next_steps = action_data.get('next_steps', [])
    next_steps_str = "\n".join([f"  {i}. {s}" for i, s in enumerate(next_steps, 1)]) or "  None specified"
    
    notes_content = f"""╔══════════════════════════════════════════════════════════════════════╗
║           MinutesX - AI MEETING NOTES (Multi-Agent System)           ║
╚══════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Meeting ID: {result.get('meeting_id', 'N/A')}
Model: Gemini 2.5 Flash (Multi-Agent Architecture)
Transcript Length: {result.get('transcript_length', 0)} characters

═══════════════════════════════════════════════════════════════════════

## MEETING OVERVIEW

{summary_data.get('executive_summary', 'See summary file for details.')}

═══════════════════════════════════════════════════════════════════════

## ACTION ITEMS
{actions_str}
═══════════════════════════════════════════════════════════════════════

## FOLLOW-UP REQUIRED

{follow_ups_str}

═══════════════════════════════════════════════════════════════════════

## NEXT STEPS

{next_steps_str}

═══════════════════════════════════════════════════════════════════════

## KEY DECISIONS

{decisions_str}

═══════════════════════════════════════════════════════════════════════

## MULTI-AGENT SYSTEM INFO

Architecture: Orchestrator + 3 Parallel Agents
  • CaptionAgent  → Headlines & social captions
  • SummaryAgent  → Executive summary & key points  
  • ActionAgent   → Action items & follow-ups

Protocol: A2A (Agent-to-Agent)
Messages Exchanged: {result.get('a2a_stats', {}).get('total_messages', 0)}
Active Agents: {', '.join(result.get('a2a_stats', {}).get('agents', []))}

═══════════════════════════════════════════════════════════════════════

AGENT: ActionAgent + Orchestrator
A2A PROTOCOL: Multi-agent coordination complete

Powered by MinutesX | github.com/harshbopaliya/MinutesX
"""
    notes_file.write_text(notes_content, encoding='utf-8')
    
    # 4. Raw transcript
    transcript_file = OUTPUT_DIR / f"transcript_{timestamp}.txt"
    transcript_file.write_text(
        f"Meeting Transcript - {datetime.now().strftime('%B %d, %Y')}\n"
        f"Meeting ID: {result.get('meeting_id', 'N/A')}\n\n"
        f"{transcript}",
        encoding='utf-8'
    )
    
    return {
        'caption': caption_file,
        'summary': summary_file,
        'notes': notes_file,
        'transcript': transcript_file
    }


def display_results(result: Dict[str, Any]):
    """Display results in console."""
    caption = result.get("caption", {})
    summary = result.get("summary", {})
    actions = result.get("actions", {})
    
    console.print("\n" + "="*70)
    
    # Caption
    console.print(Panel(
        f"[bold cyan]{caption.get('headline', 'Meeting Summary')}[/bold cyan]\n\n"
        f"[dim]{caption.get('one_liner', '')}[/dim]",
        title="📝 AI CAPTION",
        border_style="cyan"
    ))
    
    # Summary
    console.print(Panel(
        f"{summary.get('executive_summary', 'N/A')[:500]}",
        title="📋 AI SUMMARY",
        border_style="green"
    ))
    
    # Action Items
    action_items = actions.get('action_items', [])
    if action_items:
        table = Table(title="✅ ACTION ITEMS", show_header=True, header_style="bold magenta")
        table.add_column("P", width=3)
        table.add_column("Task", width=40)
        table.add_column("Owner", width=15)
        table.add_column("Due", width=10)
        
        for item in action_items[:7]:
            p = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.get('priority', 'medium'), "⚪")
            table.add_row(p, item.get('task', '')[:40], item.get('owner', 'TBD'), item.get('due', 'TBD'))
        
        console.print(table)


# =============================================================================
# Main Demo Functions
# =============================================================================

def run_live_demo():
    """Run LIVE Google Meet capture with multi-agent processing."""
    console.print(Panel.fit(
        "[bold cyan]🎤 LIVE MODE - Google Meet + Multi-Agent AI[/bold cyan]\n\n"
        "• Captures system audio from Google Meet\n"
        "• Transcribes with Gemini 2.5 Flash\n"
        "• Processes through 4-agent pipeline:\n"
        "  [dim]Orchestrator → Caption + Summary + Action Agents[/dim]\n\n"
        "[yellow]Press Ctrl+C to stop and generate notes[/yellow]",
        title="MinutesX Multi-Agent System",
        border_style="cyan"
    ))
    
    # Initialize orchestrator with all agents
    orchestrator = OrchestratorAgent()
    capture = AudioCapture(chunk_seconds=8)
    
    # Show audio devices
    console.print("\n[bold cyan]📢 Audio Devices:[/bold cyan]")
    devices = capture.list_devices()
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=4)
    table.add_column("Device", width=50)
    table.add_column("Type", width=12)
    
    for d in devices:
        dtype = "[green]LOOPBACK[/green]" if d['loopback'] else "[dim]Mic[/dim]"
        table.add_row(str(d['idx']), d['name'], dtype)
    
    console.print(table)
    
    loopback = capture.find_loopback()
    if loopback:
        console.print(f"\n[green]✓ Auto-detected loopback: [{loopback}][/green]")
    else:
        console.print("\n[yellow]⚠ No loopback found. Enable Stereo Mix in Windows Sound settings.[/yellow]")
    
    console.print("\n[cyan]Enter device number (or Enter for auto):[/cyan] ", end="")
    try:
        inp = input().strip()
        device = int(inp) if inp else loopback
    except:
        device = loopback
    
    if not capture.start(device):
        console.print("[red]Failed to start audio. Running demo mode.[/red]")
        return run_demo_mode()
    
    console.print("\n" + "="*60)
    console.print("[bold red]🔴 RECORDING[/bold red] - Listening to Google Meet...")
    console.print("[yellow]Press Ctrl+C when meeting ends[/yellow]")
    console.print("="*60 + "\n")
    
    transcript_parts = []
    
    console.print("[bold]Live Captions:[/bold]\n")
    
    try:
        while True:
            audio = capture.get_chunk()
            if audio is not None and len(audio) > 0:
                wav = capture.to_wav(audio)
                duration = capture.get_duration_str()
                
                console.print(f"[dim]⏱ {duration}[/dim] ", end="")
                
                text = orchestrator.transcription_agent.transcribe(wav)
                if text:
                    transcript_parts.append(text)
                    display = text[:100] + "..." if len(text) > 100 else text
                    console.print(f"[cyan]{display}[/cyan]")
                else:
                    console.print("[dim](listening...)[/dim]")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⏹ Stopping...[/yellow]")
    
    capture.stop()
    transcript = "\n".join(transcript_parts)
    
    if not transcript.strip():
        console.print("[yellow]No speech detected. Running demo mode.[/yellow]")
        return run_demo_mode()
    
    console.print(f"\n[green]✓ Captured {len(transcript)} characters[/green]")
    
    # Process through multi-agent system
    result = orchestrator.process_meeting(transcript)
    
    # Save and display
    files = save_outputs(result, transcript)
    display_results(result)
    show_output_summary(files, result)


def run_demo_mode():
    """Run demo with sample transcript through multi-agent system."""
    console.print(Panel.fit(
        "[bold cyan]📄 DEMO MODE - Multi-Agent Processing[/bold cyan]\n\n"
        "Processing sample meeting transcript\n"
        "through the multi-agent pipeline",
        title="MinutesX Demo",
        border_style="yellow"
    ))
    
    # Load transcript
    sample_file = Path("demo_transcript.txt")
    if sample_file.exists():
        transcript = sample_file.read_text(encoding='utf-8')
        console.print(f"\n[green]✓ Loaded: {sample_file}[/green]")
    else:
        transcript = get_sample_transcript()
        console.print("\n[yellow]Using built-in sample transcript[/yellow]")
    
    # Initialize and run multi-agent system
    console.print("\n[bold]Initializing Multi-Agent System...[/bold]")
    orchestrator = OrchestratorAgent()
    
    # Process meeting
    result = orchestrator.process_meeting(transcript)
    
    # Save and display
    files = save_outputs(result, transcript)
    display_results(result)
    show_output_summary(files, result)


def get_sample_transcript():
    """Get built-in sample transcript."""
    return """
Sarah Chen: Good morning everyone. Let's start our Q1 planning meeting. We have important items to discuss.

John Martinez: Thanks Sarah. Looking at our metrics, the Android crash rate is at 2.5%, above our 1% target. This needs immediate attention.

Mike Thompson: I've investigated - it's memory issues on older devices. I can fix it by January 15th if we prioritize it.

Sarah Chen: Mike, that's your top priority. What about user onboarding?

Lisa Wang: Our data shows 40% drop-off at step 3. I'm proposing a simplified 3-step wizard.

David Kim: Users who complete onboarding have 3x higher retention. This is a major opportunity.

Sarah Chen: Lisa, prepare mockups by January 10th. David, I need a funnel report by Thursday.

John Martinez: I also want to allocate 20% of sprint capacity to technical debt.

Sarah Chen: Approved. Let's discuss the API initiative for enterprise customers.

Mike Thompson: We can have a beta API ready by end of March - about 6 weeks of work.

Sarah Chen: John, assign Priya to lead API development. Let me summarize:
- Mike: Fix Android crashes by January 15
- Lisa: Onboarding mockups by January 10
- David: Funnel report by Thursday
- John: 20% to tech debt, assign Priya to API

John Martinez: I'll set up weekly syncs on Tuesdays.

Sarah Chen: Great meeting everyone. Let's execute and reconvene in two weeks.
"""


def show_output_summary(files: Dict, result: Dict):
    """Show final output summary."""
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        f"[bold green]✅ MULTI-AGENT PROCESSING COMPLETE![/bold green]\n\n"
        f"📁 Output Location: [cyan]{OUTPUT_DIR.absolute()}[/cyan]\n\n"
        f"📝 [bold]ai_caption.txt[/bold]  - Headlines & social captions\n"
        f"📋 [bold]ai_summary.txt[/bold]  - Executive summary & key points\n"
        f"📄 [bold]ai_notes.txt[/bold]    - Action items & follow-ups\n"
        f"📜 [dim]{files['transcript'].name}[/dim] - Raw transcript\n\n"
        f"[dim]─────────────────────────────────────────[/dim]\n"
        f"🤖 Agents: Orchestrator + Caption + Summary + Action\n"
        f"📨 A2A Messages: {result.get('a2a_stats', {}).get('total_messages', 0)}\n"
        f"⏱ Processing: {result.get('duration_seconds', 0)}s",
        title="Files Generated",
        border_style="green"
    ))


def main():
    """Main entry point."""
    console.print(Panel.fit(
        "[bold magenta]╔═══════════════════════════════════════════════════════╗[/bold magenta]\n"
        "[bold magenta]║       MinutesX - AI Meeting Notes (Multi-Agent)       ║[/bold magenta]\n"
        "[bold magenta]║           Powered by Gemini 2.5 Flash                 ║[/bold magenta]\n"
        "[bold magenta]╚═══════════════════════════════════════════════════════╝[/bold magenta]\n\n"
        "[white]Transform Google Meet calls into actionable notes[/white]\n"
        "[white]using a multi-agent AI system with A2A protocol.[/white]\n\n"
        "[dim]Agents: Orchestrator → Caption + Summary + Action[/dim]",
        border_style="magenta"
    ))
    
    console.print("\n[bold cyan]Select Mode:[/bold cyan]")
    console.print("  [bold green][1][/bold green] 🎤 [bold]LIVE[/bold] - Capture Google Meet audio")
    console.print("  [bold yellow][2][/bold yellow] 📄 [bold]DEMO[/bold] - Process sample transcript")
    
    console.print("\n[cyan]Enter 1 or 2:[/cyan] ", end="")
    
    try:
        choice = input().strip()
        
        if choice == "1":
            if AUDIO_OK:
                run_live_demo()
            else:
                console.print("\n[red]❌ Audio libraries not installed![/red]")
                console.print("[yellow]Run: pip install sounddevice numpy[/yellow]")
                console.print("\n[dim]Running demo mode...[/dim]\n")
                time.sleep(1)
                run_demo_mode()
        else:
            run_demo_mode()
            
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Cancelled.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
