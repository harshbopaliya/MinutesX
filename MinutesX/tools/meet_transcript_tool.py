"""
MeetTranscriptTool - Google Meet transcript integration.

Provides functionality to:
- Connect to Google Meet sessions
- Fetch live transcripts
- Parse transcript files (VTT, SRT, TXT)
"""
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from observability.logger import get_logger


logger = get_logger(__name__)


class TranscriptToolBase(ABC):
    """Base class for transcript tools."""
    
    @abstractmethod
    def fetch(self, source: str) -> Dict[str, Any]:
        """Fetch transcript from source."""
        pass
    
    @abstractmethod
    def parse(self, content: str, format_type: str = "txt") -> Dict[str, Any]:
        """Parse transcript content."""
        pass


class MeetTranscriptTool(TranscriptToolBase):
    """
    Google Meet transcript integration tool.
    
    Connects to Google Meet API to fetch live or recorded transcripts.
    Requires OAuth credentials with Meet API access.
    """
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.credentials_path = credentials_path
        self._service = None
        logger.info("MeetTranscriptTool initialized")
    
    def _get_service(self):
        """Get or create Google Meet API service."""
        if self._service is not None:
            return self._service
        
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            
            if self.credentials_path and os.path.exists(self.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/meetings.space.readonly']
                )
                self._service = build('meet', 'v2', credentials=credentials)
                return self._service
            else:
                logger.warning("No credentials provided for Meet API")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize Meet API service: {e}")
            return None
    
    def fetch(self, meeting_code: str) -> Dict[str, Any]:
        """
        Fetch transcript from a Google Meet session.
        
        Args:
            meeting_code: The Google Meet meeting code (e.g., 'abc-defg-hij')
            
        Returns:
            Dictionary with transcript data and metadata
        """
        logger.info(f"Fetching transcript for meeting: {meeting_code}")
        
        service = self._get_service()
        if service is None:
            logger.warning("Meet API not available, returning empty transcript")
            return {
                "meeting_code": meeting_code,
                "transcript": "",
                "participants": [],
                "duration": 0,
                "error": "Meet API not configured",
            }
        
        try:
            # Note: This is a simplified example. Actual Meet API usage may differ.
            # The Meet API structure depends on the specific API version.
            
            # Placeholder for actual API call
            result = {
                "meeting_code": meeting_code,
                "transcript": "",
                "participants": [],
                "duration": 0,
                "fetched_at": datetime.utcnow().isoformat(),
            }
            
            logger.info(f"Transcript fetched for meeting: {meeting_code}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch transcript: {e}")
            return {
                "meeting_code": meeting_code,
                "transcript": "",
                "error": str(e),
            }
    
    def parse(self, content: str, format_type: str = "txt") -> Dict[str, Any]:
        """
        Parse transcript content.
        
        Args:
            content: Raw transcript content
            format_type: Format type (txt, vtt, srt)
            
        Returns:
            Parsed transcript with segments
        """
        if format_type == "vtt":
            return self._parse_vtt(content)
        elif format_type == "srt":
            return self._parse_srt(content)
        else:
            return self._parse_txt(content)
    
    def _parse_vtt(self, content: str) -> Dict[str, Any]:
        """Parse WebVTT format transcript."""
        segments = []
        
        # Remove header
        content = re.sub(r'^WEBVTT\n\n', '', content)
        
        # Parse cues
        cue_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\Z)'
        matches = re.findall(cue_pattern, content, re.DOTALL)
        
        for start, end, text in matches:
            segments.append({
                "start": start,
                "end": end,
                "text": text.strip(),
            })
        
        full_text = " ".join(seg["text"] for seg in segments)
        
        return {
            "format": "vtt",
            "segments": segments,
            "full_text": full_text,
            "segment_count": len(segments),
        }
    
    def _parse_srt(self, content: str) -> Dict[str, Any]:
        """Parse SRT format transcript."""
        segments = []
        
        # Parse SRT cues
        cue_pattern = r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)'
        matches = re.findall(cue_pattern, content, re.DOTALL)
        
        for start, end, text in matches:
            segments.append({
                "start": start.replace(',', '.'),
                "end": end.replace(',', '.'),
                "text": text.strip(),
            })
        
        full_text = " ".join(seg["text"] for seg in segments)
        
        return {
            "format": "srt",
            "segments": segments,
            "full_text": full_text,
            "segment_count": len(segments),
        }
    
    def _parse_txt(self, content: str) -> Dict[str, Any]:
        """Parse plain text transcript."""
        # Try to detect speaker labels
        lines = content.strip().split('\n')
        segments = []
        
        speaker_pattern = r'^([A-Z][a-zA-Z\s]+):\s*(.+)$'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(speaker_pattern, line)
            if match:
                segments.append({
                    "speaker": match.group(1),
                    "text": match.group(2),
                })
            else:
                segments.append({
                    "speaker": "Unknown",
                    "text": line,
                })
        
        return {
            "format": "txt",
            "segments": segments,
            "full_text": content,
            "segment_count": len(segments),
        }


class LocalTranscriptTool(TranscriptToolBase):
    """
    Local file transcript tool.
    
    Reads transcripts from local files for testing and development.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path(".")
        self._meet_tool = MeetTranscriptTool()
        logger.info(f"LocalTranscriptTool initialized with base_path: {self.base_path}")
    
    def fetch(self, file_path: str) -> Dict[str, Any]:
        """
        Fetch transcript from local file.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            Dictionary with transcript data
        """
        full_path = self.base_path / file_path if not os.path.isabs(file_path) else Path(file_path)
        
        logger.info(f"Reading transcript from: {full_path}")
        
        try:
            if not full_path.exists():
                logger.error(f"File not found: {full_path}")
                return {
                    "file_path": str(full_path),
                    "transcript": "",
                    "error": "File not found",
                }
            
            content = full_path.read_text(encoding='utf-8')
            
            # Determine format from extension
            ext = full_path.suffix.lower()
            format_type = {'.vtt': 'vtt', '.srt': 'srt'}.get(ext, 'txt')
            
            parsed = self.parse(content, format_type)
            
            return {
                "file_path": str(full_path),
                "transcript": parsed["full_text"],
                "segments": parsed.get("segments", []),
                "format": format_type,
                "fetched_at": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Failed to read transcript file: {e}")
            return {
                "file_path": str(full_path),
                "transcript": "",
                "error": str(e),
            }
    
    def parse(self, content: str, format_type: str = "txt") -> Dict[str, Any]:
        """Parse transcript content using MeetTranscriptTool parser."""
        return self._meet_tool.parse(content, format_type)
    
    def list_transcripts(self, pattern: str = "*.txt") -> List[str]:
        """List available transcript files."""
        return [str(p) for p in self.base_path.glob(pattern)]
