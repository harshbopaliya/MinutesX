"""
CaptionAgent - Social/business caption generation agent.

Generates shareable captions suitable for:
- LinkedIn posts
- Slack updates
- Email subject lines
- Internal newsletters
"""
import json
import re
from typing import Any, Dict, Optional

import google.generativeai as genai

from observability.logger import get_logger
from observability.tracer import trace_span


logger = get_logger(__name__)


class CaptionAgent:
    """
    Parallel agent for generating shareable meeting captions.
    
    Creates captions optimized for different platforms and use cases.
    """
    
    CAPTION_PROMPT = """You are an expert at creating engaging, professional captions for meetings.

MEETING TRANSCRIPT:
{transcript}

Generate captions suitable for different use cases. Each caption should:
- Be concise and impactful
- Highlight the main outcome or theme
- Be professional and shareable

Return a JSON object:
{{
    "headline": "A punchy headline (max 80 chars)",
    "one_liner": "One-line summary (max 140 chars) suitable for Twitter/X",
    "linkedin": "Professional LinkedIn-style caption (2-3 sentences)",
    "slack": "Casual Slack update (1-2 sentences)",
    "email_subject": "Email subject line (max 60 chars)",
    "newsletter": "Brief newsletter blurb (3-4 sentences)",
    "hashtags": ["relevant", "hashtags", "without", "hash", "symbol"]
}}

Make each caption engaging and action-oriented.
Respond ONLY with the JSON object."""

    def __init__(self, model: genai.GenerativeModel):
        self.model = model
        logger.info("CaptionAgent initialized")
    
    @trace_span("caption_agent.generate_caption")
    def generate_caption(
        self,
        transcript: str,
        style: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate shareable captions from transcript.
        
        Args:
            transcript: The meeting transcript
            style: Optional preferred style (professional, casual, technical)
            
        Returns:
            Dictionary with various caption formats
        """
        logger.debug("Generating captions", extra={"transcript_length": len(transcript)})
        
        prompt = self.CAPTION_PROMPT.format(
            transcript=self._extract_key_parts(transcript),
        )
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_response(response.text)
            
            logger.info("Captions generated successfully", extra={
                "headline_length": len(result.get("headline", "")),
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return self._fallback_caption(transcript)
    
    def _extract_key_parts(self, transcript: str, max_chars: int = 10000) -> str:
        """Extract the most relevant parts of the transcript for caption generation."""
        if len(transcript) <= max_chars:
            return transcript
        
        # Take beginning (context), middle (core discussion), and end (conclusions)
        third = max_chars // 3
        return (
            f"{transcript[:third]}\n\n"
            f"[...]\n\n"
            f"{transcript[len(transcript)//2 - third//2:len(transcript)//2 + third//2]}\n\n"
            f"[...]\n\n"
            f"{transcript[-third:]}"
        )
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from model."""
        text = response_text.strip()
        
        if text.startswith("```"):
            text = re.sub(r"```json?\n?", "", text)
            text = re.sub(r"```\n?$", "", text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse caption JSON")
            return self._fallback_caption("")
    
    def _fallback_caption(self, transcript: str) -> Dict[str, Any]:
        """Generate fallback captions if LLM fails."""
        words = transcript.split()[:20]
        preview = " ".join(words)
        
        return {
            "headline": "Meeting Summary",
            "one_liner": f"Meeting notes: {preview}..." if words else "Meeting completed",
            "linkedin": "Just wrapped up an important team meeting. Stay tuned for updates!",
            "slack": "Meeting completed - notes coming soon!",
            "email_subject": "Meeting Notes Available",
            "newsletter": "The team met to discuss important topics. Full notes are available for review.",
            "hashtags": ["meeting", "teamwork", "productivity"],
        }
    
    def format_with_hashtags(self, caption: str, hashtags: list) -> str:
        """Format caption with hashtags for social media."""
        tags = " ".join(f"#{tag}" for tag in hashtags[:5])
        return f"{caption}\n\n{tags}"
