"""
SpeakerIdentifierTool - Speaker diarization and identification.

Identifies and tracks speakers throughout a meeting transcript.
"""
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from observability.logger import get_logger


logger = get_logger(__name__)


class SpeakerIdentifierTool:
    """
    Tool for identifying and tracking speakers in meeting transcripts.
    
    Features:
    - Extract speaker names from formatted transcripts
    - Track speaking time per participant
    - Identify speaker turns and patterns
    """
    
    def __init__(self):
        logger.info("SpeakerIdentifierTool initialized")
    
    def identify_speakers(self, transcript: str) -> Dict[str, Any]:
        """
        Identify speakers in a transcript.
        
        Args:
            transcript: The meeting transcript text
            
        Returns:
            Dictionary with speakers and their statistics
        """
        logger.debug("Identifying speakers in transcript")
        
        speakers = defaultdict(lambda: {"turns": 0, "words": 0, "segments": []})
        
        # Pattern for "Speaker Name: text" format
        pattern = r'^([A-Z][a-zA-Z\s]+):\s*(.+)$'
        
        lines = transcript.strip().split('\n')
        current_speaker = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                speaker_name = match.group(1).strip()
                text = match.group(2).strip()
                
                if speaker_name != current_speaker:
                    speakers[speaker_name]["turns"] += 1
                    current_speaker = speaker_name
                
                word_count = len(text.split())
                speakers[speaker_name]["words"] += word_count
                speakers[speaker_name]["segments"].append({
                    "line": i + 1,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "word_count": word_count,
                })
        
        # Calculate statistics
        total_words = sum(s["words"] for s in speakers.values())
        speaker_stats = {}
        
        for name, data in speakers.items():
            speaker_stats[name] = {
                "turns": data["turns"],
                "words": data["words"],
                "percentage": round(data["words"] / total_words * 100, 1) if total_words > 0 else 0,
                "avg_words_per_turn": round(data["words"] / data["turns"], 1) if data["turns"] > 0 else 0,
            }
        
        result = {
            "speakers": list(speakers.keys()),
            "speaker_count": len(speakers),
            "total_words": total_words,
            "statistics": speaker_stats,
        }
        
        logger.info(f"Identified {len(speakers)} speakers")
        return result
    
    def extract_speaker_segments(
        self,
        transcript: str,
        speaker_name: str,
    ) -> List[str]:
        """
        Extract all segments spoken by a specific speaker.
        
        Args:
            transcript: The meeting transcript
            speaker_name: Name of the speaker to extract
            
        Returns:
            List of text segments from that speaker
        """
        segments = []
        pattern = rf'^{re.escape(speaker_name)}:\s*(.+)$'
        
        for line in transcript.split('\n'):
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                segments.append(match.group(1).strip())
        
        return segments
    
    def get_conversation_flow(self, transcript: str) -> List[Dict[str, str]]:
        """
        Get the flow of conversation with speaker turns.
        
        Args:
            transcript: The meeting transcript
            
        Returns:
            List of speaker turns in order
        """
        flow = []
        pattern = r'^([A-Z][a-zA-Z\s]+):\s*(.+)$'
        
        for line in transcript.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                flow.append({
                    "speaker": match.group(1).strip(),
                    "text": match.group(2).strip(),
                })
        
        return flow
    
    def find_mentions(
        self,
        transcript: str,
        name: str,
    ) -> List[Dict[str, Any]]:
        """
        Find all mentions of a person's name in the transcript.
        
        Args:
            transcript: The meeting transcript
            name: Name to search for
            
        Returns:
            List of mentions with context
        """
        mentions = []
        lines = transcript.split('\n')
        
        for i, line in enumerate(lines):
            if name.lower() in line.lower():
                mentions.append({
                    "line": i + 1,
                    "context": line.strip(),
                    "is_speaker": line.strip().startswith(name),
                })
        
        return mentions
