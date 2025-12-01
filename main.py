"""
MinutesX - Intelligent Meeting Notes with Gemini 2.5 Flash

A multi-agent system for real-time meeting transcription, summarization,
and action item extraction using Google Gemini AI.

Features:
- Live audio capture from Google Meet (system audio loopback)
- Real-time transcription and captions in English
- Meeting summarization with key points
- Action item extraction
- Multi-agent processing pipeline
"""
import asyncio
import os
import sys
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

from config import config, get_config
from observability.logger import get_logger


logger = get_logger(__name__)
console = Console()


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    console.print("\n[yellow]Shutdown requested. Finishing up...[/yellow]")


@click.group()
@click.version_option(version="0.1.0", prog_name="MinutesX")
def cli():
    """MinutesX - Intelligent Meeting Notes with Gemini 2.5 Flash"""
    pass


@cli.command()
@click.option("--source", "-s", default="system", help="Audio source: system, microphone, or file path")
@click.option("--output", "-o", default="./meeting_notes", help="Output directory for meeting notes")
@click.option("--duration", "-d", default=0, type=int, help="Max duration in minutes (0 = unlimited)")
@click.option("--show-captions/--no-captions", default=True, help="Show live captions")
def listen(source: str, output: str, duration: int, show_captions: bool):
    """
    Start listening to a meeting and generate live transcription.
    
    This captures audio from Google Meet (or other sources) and provides:
    - Real-time captions in English
    - Live transcript accumulation
    - Press Ctrl+C to stop and generate meeting notes
    
    Examples:
        minutesx listen                    # Listen to system audio (Google Meet)
        minutesx listen -s microphone      # Listen to microphone only
        minutesx listen -s recording.wav   # Process an audio file
    """
    console.print(Panel.fit(
        "[bold cyan]MinutesX - Live Meeting Transcription[/bold cyan]\n\n"
        f"Audio Source: [green]{source}[/green]\n"
        f"Output: [green]{output}[/green]\n"
        f"Duration: [green]{'Unlimited' if duration == 0 else f'{duration} minutes'}[/green]\n\n"
        "[yellow]Press Ctrl+C to stop and generate meeting notes[/yellow]",
        title="Starting..."
    ))
    
    # Validate API key
    if not config.gemini.api_key:
        console.print("[red]Error: GOOGLE_API_KEY not set. Please set it in .env file.[/red]")
        sys.exit(1)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != 'win32':
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the async listener
    asyncio.run(_run_listener(source, output, duration, show_captions))


async def _run_listener(source: str, output: str, duration: int, show_captions: bool):
    """Run the live meeting listener."""
    global _shutdown_requested
    
    try:
        from tools.audio_capture import create_audio_capture
        from tools.live_transcription import LiveTranscriptionService, TranscriptionProcessor
        
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        audio_capture = create_audio_capture(source)
        transcription_service = LiveTranscriptionService()
        
        # Caption display callback
        def on_caption(caption):
            if show_captions:
                speaker = f"[{caption.speaker}] " if caption.speaker else ""
                console.print(f"[cyan]>>> {speaker}{caption.text}[/cyan]")
        
        def on_transcript(segment):
            # Log transcript segment (could also save incrementally)
            pass
        
        transcription_service.add_caption_callback(on_caption)
        transcription_service.add_transcript_callback(on_transcript)
        
        console.print("\n[green]✓ Starting audio capture...[/green]")
        console.print("[dim]Listening for meeting audio. Speak or play Google Meet...[/dim]\n")
        
        # Start transcription
        start_time = datetime.now()
        max_duration = duration * 60 if duration > 0 else float('inf')
        
        audio_capture.start()
        
        while not _shutdown_requested:
            # Check duration limit
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= max_duration:
                console.print("\n[yellow]Duration limit reached.[/yellow]")
                break
            
            # Get and process audio chunk
            chunk = audio_capture.get_audio_chunk(timeout=10.0)
            
            if chunk:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    transcription_service.transcribe_audio_chunk,
                    chunk,
                )
            
            await asyncio.sleep(0.1)
        
        # Stop capture
        audio_capture.stop()
        transcription_service.stop()
        
        # Get full transcript
        full_transcript = transcription_service.get_full_transcript()
        
        if not full_transcript.strip():
            console.print("\n[yellow]No speech detected in the recording.[/yellow]")
            return
        
        console.print("\n[green]✓ Meeting ended. Processing notes...[/green]\n")
        
        # Process transcript through agents
        await _process_and_save_meeting(full_transcript, output_path, start_time)
        
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Run: pip install sounddevice numpy[/yellow]")
    except Exception as e:
        logger.exception("Listener error")
        console.print(f"[red]Error: {e}[/red]")


async def _process_and_save_meeting(transcript: str, output_path: Path, start_time: datetime):
    """Process transcript and save meeting notes."""
    from agents.orchestrator_agent import MeetingOrchestratorAgent
    import uuid
    
    meeting_id = f"meeting_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    console.print("[cyan]Processing with AI agents...[/cyan]")
    
    with console.status("[bold green]Analyzing meeting content..."):
        try:
            orchestrator = MeetingOrchestratorAgent()
            result = orchestrator.process_meeting(
                meeting_id=meeting_id,
                transcript=transcript,
            )
            
            # Save outputs
            _save_meeting_outputs(result, transcript, output_path, meeting_id)
            
            # Display summary
            _display_meeting_summary(result)
            
        except Exception as e:
            logger.exception("Processing error")
            console.print(f"[red]Processing failed: {e}[/red]")
            
            # Save raw transcript at least
            transcript_file = output_path / f"{meeting_id}_transcript.txt"
            transcript_file.write_text(transcript)
            console.print(f"[yellow]Raw transcript saved to: {transcript_file}[/yellow]")


def _save_meeting_outputs(result, transcript: str, output_path: Path, meeting_id: str):
    """Save all meeting outputs to files."""
    import json
    
    # Save transcript
    transcript_file = output_path / f"{meeting_id}_transcript.txt"
    transcript_file.write_text(transcript)
    
    # Save full results as JSON
    json_file = output_path / f"{meeting_id}_results.json"
    json_file.write_text(json.dumps(result.to_dict(), indent=2))
    
    # Save formatted meeting notes as Markdown
    md_content = _format_meeting_notes_md(result)
    md_file = output_path / f"{meeting_id}_notes.md"
    md_file.write_text(md_content)
    
    console.print(f"\n[green]✓ Meeting notes saved:[/green]")
    console.print(f"  • Transcript: {transcript_file}")
    console.print(f"  • Full results: {json_file}")
    console.print(f"  • Meeting notes: {md_file}")


def _format_meeting_notes_md(result) -> str:
    """Format meeting results as Markdown."""
    lines = [
        f"# Meeting Notes",
        f"",
        f"**Date:** {result.processed_at.strftime('%B %d, %Y at %I:%M %p')}",
        f"**Category:** {result.category}",
        f"",
        f"## Summary",
        f"",
        f"**One-liner:** {result.caption}",
        f"",
    ]
    
    # Executive summary
    if result.summary.get("executive_summary"):
        lines.extend([
            f"{result.summary['executive_summary']}",
            f"",
        ])
    
    # Key Points
    if result.summary.get("key_points"):
        lines.extend([
            f"## Key Points",
            f"",
        ])
        for point in result.summary["key_points"]:
            lines.append(f"- {point}")
        lines.append("")
    
    # Decisions
    if result.decisions:
        lines.extend([
            f"## Decisions Made",
            f"",
        ])
        for decision in result.decisions:
            lines.append(f"- {decision}")
        lines.append("")
    
    # Action Items
    if result.action_items:
        lines.extend([
            f"## Action Items",
            f"",
        ])
        for item in result.action_items:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                item.get("priority", "medium"), "⚪"
            )
            owner = item.get("owner", "Unassigned")
            due = f" (Due: {item['due_date']})" if item.get("due_date") else ""
            lines.append(f"- {priority_emoji} **{item['description']}**")
            lines.append(f"  - Owner: {owner}{due}")
        lines.append("")
    
    # Key Topics
    if result.key_topics:
        lines.extend([
            f"## Topics Discussed",
            f"",
            f"{', '.join(result.key_topics)}",
            f"",
        ])
    
    return "\n".join(lines)


def _display_meeting_summary(result):
    """Display meeting summary in console."""
    console.print("\n")
    console.print(Panel(
        f"[bold]{result.caption}[/bold]",
        title="📝 Meeting Caption",
        border_style="cyan"
    ))
    
    # Summary
    if result.summary.get("executive_summary"):
        console.print(Panel(
            result.summary["executive_summary"],
            title="📋 Executive Summary",
            border_style="green"
        ))
    
    # Action Items Table
    if result.action_items:
        table = Table(title="✅ Action Items", show_header=True, header_style="bold magenta")
        table.add_column("Priority", style="cyan", width=8)
        table.add_column("Task", style="white")
        table.add_column("Owner", style="yellow")
        table.add_column("Due", style="green")
        
        for item in result.action_items:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                item.get("priority", "medium"), "⚪"
            )
            table.add_row(
                priority_emoji,
                item.get("description", "")[:50],
                item.get("owner", "Unassigned"),
                item.get("due_date") or "-"
            )
        
        console.print(table)
    
    # Key Decisions
    if result.decisions:
        console.print("\n[bold]🎯 Key Decisions:[/bold]")
        for decision in result.decisions[:5]:
            console.print(f"  • {decision}")


@cli.command()
@click.argument("transcript_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="./meeting_notes", help="Output directory")
def process(transcript_file: str, output: str):
    """
    Process an existing transcript file and generate meeting notes.
    
    Example:
        minutesx process ./transcript.txt
    """
    console.print(f"[cyan]Processing transcript: {transcript_file}[/cyan]")
    
    if not config.gemini.api_key:
        console.print("[red]Error: GOOGLE_API_KEY not set.[/red]")
        sys.exit(1)
    
    # Read transcript
    transcript = Path(transcript_file).read_text()
    
    if not transcript.strip():
        console.print("[red]Error: Transcript file is empty.[/red]")
        sys.exit(1)
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    asyncio.run(_process_and_save_meeting(
        transcript, 
        output_path, 
        datetime.now()
    ))


@cli.command()
@click.option("--device", "-d", default=None, type=int, help="Device index to test")
def test_audio(device: Optional[int]):
    """
    Test audio capture and list available devices.
    
    Use this to verify your system audio setup for Google Meet capture.
    """
    console.print("[cyan]Testing Audio Capture...[/cyan]\n")
    
    try:
        from tools.audio_capture import SystemAudioCapture
        
        capture = SystemAudioCapture(device_index=device)
        
        # List devices
        devices = capture.list_devices()
        
        table = Table(title="Audio Devices", show_header=True, header_style="bold")
        table.add_column("Index", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Channels", style="green")
        table.add_column("Loopback", style="yellow")
        
        for dev in devices:
            loopback = "✓" if dev.get("is_loopback") else ""
            table.add_row(
                str(dev["index"]),
                dev["name"][:50],
                str(dev["channels"]),
                loopback
            )
        
        console.print(table)
        
        # Find loopback
        loopback_idx = capture.find_loopback_device()
        if loopback_idx is not None:
            console.print(f"\n[green]✓ Found loopback device at index {loopback_idx}[/green]")
        else:
            console.print("\n[yellow]⚠ No loopback device found.[/yellow]")
            console.print("[dim]To capture Google Meet audio, enable 'Stereo Mix' in Windows Sound settings[/dim]")
            console.print("[dim]or use a virtual audio cable like VB-Audio Virtual Cable.[/dim]")
        
        # Test recording
        if click.confirm("\nTest 3-second audio capture?", default=True):
            console.print("[cyan]Recording for 3 seconds...[/cyan]")
            
            import sounddevice as sd
            import numpy as np
            
            test_device = device if device is not None else loopback_idx
            recording = sd.rec(
                int(16000 * 3),
                samplerate=16000,
                channels=1,
                dtype='int16',
                device=test_device
            )
            sd.wait()
            
            # Check if we got audio
            max_amplitude = np.max(np.abs(recording))
            if max_amplitude > 100:
                console.print(f"[green]✓ Audio captured! Max amplitude: {max_amplitude}[/green]")
            else:
                console.print(f"[yellow]⚠ Low audio level ({max_amplitude}). Check your audio source.[/yellow]")
                
    except ImportError:
        console.print("[red]sounddevice not installed. Run: pip install sounddevice[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def config_info():
    """Display current configuration."""
    cfg = get_config()
    
    table = Table(title="MinutesX Configuration", show_header=True, header_style="bold")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Gemini Model", cfg.gemini.model)
    table.add_row("API Key Set", "✓" if cfg.gemini.api_key else "✗")
    table.add_row("Temperature", str(cfg.gemini.temperature))
    table.add_row("Max Tokens", str(cfg.gemini.max_tokens))
    table.add_row("Memory Backend", cfg.memory.backend)
    table.add_row("Storage Path", cfg.memory.chromadb_path)
    table.add_row("Log Level", cfg.observability.log_level)
    
    console.print(table)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
