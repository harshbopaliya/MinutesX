"""
MinutesX - Live Google Meet Transcription & AI Notes
=====================================================

Captures LIVE audio from Google Meet calls and generates:
- ai_caption.txt  - One-line meeting caption
- ai_summary.txt  - Executive summary with key points  
- ai_notes.txt    - Detailed meeting notes with action items

HOW TO USE:
1. Start your Google Meet call
2. Run: python demo.py
3. Select your audio device (use Stereo Mix/WASAPI loopback)
4. Press Ctrl+C when meeting ends
5. Find your notes in ./output/ folder

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
import threading
from datetime import datetime
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.layout import Layout

console = Console()

# =============================================================================
# Configuration
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OUTPUT_DIR = Path("./output")

if not GOOGLE_API_KEY:
    console.print("[red]❌ ERROR: GOOGLE_API_KEY not set![/red]")
    console.print("\n[yellow]Add to your .env file:[/yellow]")
    console.print("GOOGLE_API_KEY=your_api_key_here")
    console.print("\n[cyan]Get your free key at: https://aistudio.google.com/apikey[/cyan]")
    sys.exit(1)

# Initialize Gemini
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# Model name - Gemini 2.0 Flash
MODEL_NAME = "gemini-2.0-flash-exp"

# Try to import audio libraries
AUDIO_OK = False
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_OK = True
except ImportError:
    console.print("[yellow]⚠ Audio libraries not installed.[/yellow]")
    console.print("[dim]Run: pip install sounddevice numpy[/dim]\n")


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
# Gemini AI Functions - Optimized for Live Transcription
# =============================================================================

def transcribe_audio(wav_bytes):
    """Transcribe audio chunk using Gemini 2.5 Flash."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        response = model.generate_content([
            """Transcribe this audio accurately. 
            - Return ONLY the spoken words
            - Include speaker names if identifiable (e.g., "John: Hello everyone")
            - If no speech detected, return [silence]
            - Keep filler words minimal""",
            {"mime_type": "audio/wav", "data": audio_b64}
        ])
        
        text = response.text.strip()
        if text and text != "[silence]" and len(text) > 3:
            return text
        return None
    except Exception as e:
        # Silent fail for transcription to not interrupt flow
        return None


def generate_caption(transcript):
    """Generate a concise meeting caption."""
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"""Create a professional one-line caption (max 100 characters) for this meeting.
Focus on the main topic, decision, or outcome.

Meeting transcript:
{transcript[:8000]}

Return ONLY the caption text, nothing else."""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return f"Meeting - {datetime.now().strftime('%B %d, %Y')}"


def generate_summary(transcript):
    """Generate comprehensive meeting summary."""
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"""Analyze this meeting transcript and create a comprehensive summary.

MEETING TRANSCRIPT:
{transcript[:15000]}

Create a detailed summary with these exact sections:

## EXECUTIVE SUMMARY
Write 3-4 sentences capturing the essence of the meeting - what was discussed, key outcomes, and overall tone.

## KEY DISCUSSION POINTS
List 5-7 main points discussed with brief context for each:
• Point 1 - brief explanation
• Point 2 - brief explanation
(continue as needed)

## DECISIONS MADE
List all decisions that were agreed upon during the meeting:
• Decision 1
• Decision 2
(if no decisions, state "No formal decisions were made")

## TOPICS COVERED
• Topic 1
• Topic 2
• Topic 3

## PARTICIPANTS
List identified speakers and their apparent roles (if determinable)

## MEETING OUTCOME
Brief paragraph on what was achieved and the overall result of the meeting.

Format with clear headers and bullet points. Be specific and actionable."""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Summary generation failed: {str(e)}"


def generate_notes(transcript):
    """Generate detailed meeting notes with action items."""
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"""Create comprehensive, professional meeting notes from this transcript.

TRANSCRIPT:
{transcript[:15000]}

Generate detailed meeting notes in this EXACT format:

## MEETING INFORMATION
- Date: {datetime.now().strftime('%B %d, %Y')}
- Time: {datetime.now().strftime('%I:%M %p')}
- Generated by: MinutesX AI (Gemini 2.5 Flash)

## MEETING OVERVIEW
Write a 2-3 paragraph summary covering what the meeting was about, who participated, and the main outcomes.

## ACTION ITEMS
Extract ALL tasks, to-dos, and commitments mentioned. Format each as:

☐ [Specific Task Description]
   → Assigned to: [Name or "Unassigned"]
   → Priority: [High/Medium/Low based on context]
   → Due: [Date if mentioned, otherwise "TBD"]

☐ [Next task...]

(Include at least 3-5 action items. If none explicitly stated, infer from commitments made)

## KEY DECISIONS
Document all decisions made during the meeting:
✓ Decision 1 - context/reasoning
✓ Decision 2 - context/reasoning

## DISCUSSION HIGHLIGHTS
Important points raised during discussion:
• Point 1 - details
• Point 2 - details
• Point 3 - details

## FOLLOW-UP REQUIRED
Items needing follow-up after this meeting:
• Follow-up item 1
• Follow-up item 2

## NEXT STEPS
What should happen after this meeting:
1. First step
2. Second step
3. Third step

## PARKING LOT
Issues raised but not resolved (for future discussion):
• Item 1
• Item 2
(or "None" if all items were addressed)

## NOTES
Any additional observations or context that might be useful.

Make notes comprehensive, professional, and immediately actionable."""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Notes generation failed: {str(e)}"


# =============================================================================
# Output Functions
# =============================================================================

def save_outputs(transcript, caption, summary, notes):
    """Save all outputs to ./output folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save AI Caption
    caption_file = OUTPUT_DIR / "ai_caption.txt"
    caption_content = f"""╔══════════════════════════════════════════════════════════════════╗
║                    MinutesX - AI CAPTION                          ║
╚══════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Model: Gemini 2.5 Flash

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEETING CAPTION:
{caption}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use this caption for:
• Slack/Teams updates
• Email subject lines
• Calendar event titles
• Meeting summaries
"""
    caption_file.write_text(caption_content, encoding='utf-8')
    
    # 2. Save AI Summary
    summary_file = OUTPUT_DIR / "ai_summary.txt"
    summary_content = f"""╔══════════════════════════════════════════════════════════════════╗
║                    MinutesX - AI SUMMARY                          ║
╚══════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Model: Gemini 2.5 Flash

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Powered by MinutesX | github.com/harshbopaliya/MinutesX
"""
    summary_file.write_text(summary_content, encoding='utf-8')
    
    # 3. Save AI Notes
    notes_file = OUTPUT_DIR / "ai_notes.txt"
    notes_content = f"""╔══════════════════════════════════════════════════════════════════╗
║                    MinutesX - AI MEETING NOTES                    ║
╚══════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Model: Gemini 2.5 Flash

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{notes}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Powered by MinutesX | github.com/harshbopaliya/MinutesX
"""
    notes_file.write_text(notes_content, encoding='utf-8')
    
    # 4. Save raw transcript
    transcript_file = OUTPUT_DIR / f"transcript_{timestamp}.txt"
    transcript_file.write_text(f"Meeting Transcript - {datetime.now().strftime('%B %d, %Y')}\n\n{transcript}", encoding='utf-8')
    
    return {
        'caption': caption_file,
        'summary': summary_file,
        'notes': notes_file,
        'transcript': transcript_file
    }


def display_results(caption, summary, notes):
    """Display results in console."""
    console.print("\n" + "="*70)
    console.print(Panel(f"[bold cyan]{caption}[/bold cyan]", title="📝 AI CAPTION", border_style="cyan"))
    
    console.print("\n")
    console.print(Panel(summary[:1500] + "..." if len(summary) > 1500 else summary, 
                       title="📋 AI SUMMARY (Preview)", border_style="green"))
    
    console.print("\n")
    console.print(Panel(notes[:1500] + "..." if len(notes) > 1500 else notes,
                       title="📄 AI NOTES (Preview)", border_style="yellow"))


# =============================================================================
# Main Demo Functions - Live Google Meet Capture
# =============================================================================

def run_live_demo():
    """Run LIVE Google Meet audio capture and transcription."""
    console.print(Panel.fit(
        "[bold cyan]🎤 LIVE MODE - Google Meet Capture[/bold cyan]\n\n"
        "• Captures system audio from Google Meet in real-time\n"
        "• Transcribes with Gemini 2.5 Flash\n"
        "• Generates ai_caption.txt, ai_summary.txt, ai_notes.txt\n\n"
        "[yellow]Press Ctrl+C to stop recording and generate notes[/yellow]",
        title="MinutesX Live",
        border_style="cyan"
    ))
    
    capture = AudioCapture(chunk_seconds=8)  # 8 second chunks
    
    # Show available audio devices
    console.print("\n[bold cyan]📢 Audio Input Devices:[/bold cyan]")
    console.print("[dim]Select a LOOPBACK device to capture Google Meet audio[/dim]\n")
    
    devices = capture.list_devices()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=4)
    table.add_column("Device Name", width=50)
    table.add_column("Type", width=15)
    
    for d in devices:
        device_type = "[green]◀ LOOPBACK[/green]" if d['loopback'] else "[dim]Microphone[/dim]"
        table.add_row(str(d['idx']), d['name'], device_type)
    
    console.print(table)
    
    # Find loopback device
    loopback = capture.find_loopback()
    if loopback is not None:
        console.print(f"\n[green]✓ Auto-detected loopback device: [{loopback}][/green]")
    else:
        console.print("\n[yellow]⚠ No loopback device detected![/yellow]")
        console.print("[dim]To capture Google Meet audio, enable 'Stereo Mix' in Windows Sound settings[/dim]")
        console.print("[dim]Or use microphone input (place near speakers)[/dim]")
    
    # Let user select device
    console.print("\n[cyan]Enter device number (or press Enter for auto):[/cyan] ", end="")
    try:
        inp = input().strip()
        device = int(inp) if inp else loopback
    except:
        device = loopback
    
    selected_name = "Default" 
    for d in devices:
        if d['idx'] == device:
            selected_name = d['name']
            break
    
    console.print(f"\n[cyan]Selected device:[/cyan] {selected_name}")
    
    # Start recording
    if not capture.start(device):
        console.print("[red]❌ Failed to start audio capture.[/red]")
        console.print("[yellow]Running demo mode instead...[/yellow]")
        return run_demo_mode()
    
    console.print("\n" + "="*60)
    console.print("[bold red]🔴 RECORDING LIVE[/bold red] - Listening to Google Meet...")
    console.print("[dim]Speak clearly or play your Google Meet meeting[/dim]")
    console.print("[yellow]Press Ctrl+C when meeting ends to generate notes[/yellow]")
    console.print("="*60 + "\n")
    
    transcript_parts = []
    chunk_count = 0
    
    console.print("[bold]Live Captions:[/bold]\n")
    
    try:
        while True:
            # Get audio chunk
            audio = capture.get_chunk()
            
            if audio is not None and len(audio) > 0:
                chunk_count += 1
                wav = capture.to_wav(audio)
                
                # Show recording indicator
                duration = capture.get_duration_str()
                console.print(f"[dim]⏱ {duration}[/dim] ", end="")
                
                # Transcribe with Gemini
                text = transcribe_audio(wav)
                
                if text:
                    transcript_parts.append(text)
                    # Show live caption
                    display_text = text[:100] + "..." if len(text) > 100 else text
                    console.print(f"[cyan]{display_text}[/cyan]")
                else:
                    console.print(f"[dim](listening...)[/dim]")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⏹ Stopping recording...[/yellow]")
    
    # Stop capture
    capture.stop()
    final_duration = capture.get_duration_str()
    
    # Combine transcript
    transcript = "\n".join(transcript_parts)
    
    if not transcript.strip():
        console.print("\n[yellow]⚠ No speech was detected during recording.[/yellow]")
        console.print("[dim]Make sure audio is playing through the selected device.[/dim]")
        console.print("\n[cyan]Running demo mode to show output format...[/cyan]")
        return run_demo_mode()
    
    console.print(f"\n[green]✓ Recording complete![/green]")
    console.print(f"  Duration: {final_duration}")
    console.print(f"  Chunks: {chunk_count}")
    console.print(f"  Transcript: {len(transcript)} characters")
    
    # Process and save
    process_and_save(transcript)


def run_demo_mode():
    """Run demo with sample transcript file."""
    console.print(Panel.fit(
        "[bold cyan]📄 DEMO MODE - Sample Transcript[/bold cyan]\n\n"
        "Processing demo_transcript.txt to show\n"
        "AI-generated meeting notes format",
        title="MinutesX Demo",
        border_style="yellow"
    ))
    
    # Load transcript from file
    sample_file = Path("demo_transcript.txt")
    if sample_file.exists():
        transcript = sample_file.read_text(encoding='utf-8')
        console.print(f"\n[green]✓ Loaded: {sample_file} ({len(transcript)} chars)[/green]")
    else:
        console.print(f"\n[yellow]⚠ demo_transcript.txt not found, using built-in sample[/yellow]")
        transcript = """
Sarah Chen: Good morning everyone. Let's start our Q1 planning meeting. We have important items to discuss today including mobile performance, user onboarding, and the API initiative.

John Martinez: Thanks Sarah. Looking at our Q4 numbers, we shipped twelve features but our Android crash rate is at 2.5%, above our 1% threshold. This is affecting our app store rating.

Mike Thompson: I've been investigating. The crashes are memory-related on older Android devices with less than 4GB RAM. I can implement a chunked sync approach to reduce memory by 60%.

Sarah Chen: How long for the fix?

Mike Thompson: By January 15th if we prioritize it. But we'd need to push the real-time collaboration feature.

John Martinez: User retention is more critical right now. Let's do it.

Sarah Chen: Agreed. Mike, crash fix is top priority. John, update the sprint plan.

Lisa Wang: Can I add something? Our onboarding has a 40% drop-off at step 3 where users connect data sources. I want to redesign it into a simpler 3-step wizard.

David Kim: Users who complete onboarding have 3x higher retention. This is a major opportunity.

Sarah Chen: Lisa, prepare mockups by January 10th. David, I need a funnel analysis report by Thursday.

John Martinez: I also want to allocate 20% of sprint capacity to technical debt. Our auth system uses deprecated libraries.

Sarah Chen: Approved. Now let's discuss the API initiative for enterprise customers.

Mike Thompson: We could have a beta API with read-only endpoints by end of March. About 6 weeks of work.

Sarah Chen: John, assign Priya to lead API development. Let me summarize action items:
- Mike: Fix Android crashes by January 15
- Lisa: Onboarding mockups by January 10
- David: Funnel report by Thursday
- John: 20% capacity to tech debt, assign Priya to API
- Weekly syncs on Tuesdays

John Martinez: I'll set up the syncs and a Thursday check-in.

Sarah Chen: Great meeting. Let's execute and reconvene in two weeks.
"""
    
    process_and_save(transcript)


def process_and_save(transcript):
    """Process transcript with Gemini and save all output files."""
    console.print(f"\n[cyan]📊 Processing transcript ({len(transcript)} characters)...[/cyan]")
    console.print("\n[bold]🤖 Generating AI outputs with Gemini 2.5 Flash...[/bold]\n")
    
    # Generate all three outputs
    with console.status("[bold green]Generating AI Caption..."):
        caption = generate_caption(transcript)
    console.print("[green]✓[/green] Caption generated")
    
    with console.status("[bold green]Generating AI Summary..."):
        summary = generate_summary(transcript)
    console.print("[green]✓[/green] Summary generated")
    
    with console.status("[bold green]Generating AI Notes with Action Items..."):
        notes = generate_notes(transcript)
    console.print("[green]✓[/green] Notes generated")
    
    # Save all outputs
    files = save_outputs(transcript, caption, summary, notes)
    
    # Display preview in console
    display_results(caption, summary, notes)
    
    # Show saved files summary
    console.print("\n" + "="*70)
    console.print(Panel.fit(
        "[bold green]✅ OUTPUT FILES GENERATED![/bold green]\n\n"
        f"📁 Location: [cyan]{OUTPUT_DIR.absolute()}[/cyan]\n\n"
        "📝 [bold]ai_caption.txt[/bold]  - One-line meeting caption\n"
        "📋 [bold]ai_summary.txt[/bold]  - Executive summary & key points\n"
        "📄 [bold]ai_notes.txt[/bold]    - Detailed notes & action items\n"
        f"📜 [dim]{files['transcript'].name}[/dim] - Raw transcript",
        title="Files Saved",
        border_style="green"
    ))
    
    console.print("\n[cyan]Open the ./output/ folder to view your meeting notes![/cyan]\n")


def main():
    """Main entry point - MinutesX Live Google Meet Notes."""
    console.print(Panel.fit(
        "[bold magenta]╔═══════════════════════════════════════════════════╗[/bold magenta]\n"
        "[bold magenta]║           MinutesX - AI Meeting Notes             ║[/bold magenta]\n"
        "[bold magenta]║         Powered by Gemini 2.5 Flash               ║[/bold magenta]\n"
        "[bold magenta]╚═══════════════════════════════════════════════════╝[/bold magenta]\n\n"
        "[white]Transform Google Meet calls into actionable[/white]\n"
        "[white]notes, summaries, and captions instantly.[/white]\n\n"
        "[dim]Output: ./output/ai_caption.txt, ai_summary.txt, ai_notes.txt[/dim]",
        border_style="magenta"
    ))
    
    console.print("\n[bold cyan]Select Mode:[/bold cyan]")
    console.print("  [bold green][1][/bold green] 🎤 [bold]LIVE MODE[/bold] - Capture audio from Google Meet")
    console.print("      [dim]→ Listens to your meeting in real-time[/dim]")
    console.print("  [bold yellow][2][/bold yellow] 📄 [bold]DEMO MODE[/bold] - Process sample transcript")
    console.print("      [dim]→ Shows output format without audio[/dim]")
    
    console.print("\n[cyan]Enter 1 or 2:[/cyan] ", end="")
    
    try:
        choice = input().strip()
        
        if choice == "1":
            if AUDIO_OK:
                run_live_demo()
            else:
                console.print("\n[red]❌ Audio libraries not installed![/red]")
                console.print("\n[yellow]To enable live mode, run:[/yellow]")
                console.print("[bold]  pip install sounddevice numpy[/bold]")
                console.print("\n[dim]Running demo mode instead...[/dim]\n")
                time.sleep(2)
                run_demo_mode()
        else:
            run_demo_mode()
            
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    main()
