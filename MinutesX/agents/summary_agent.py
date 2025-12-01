"""
SummaryAgent - Multi-level meeting summarization agent.

Generates:
- One-line caption (max 140 chars)
- Executive summary (3 sentences)
- Detailed bullet points (5-12 items)
- Key decisions made
"""
import json
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from observability.logger import get_logger
from observability.tracer import trace_span


logger = get_logger(__name__)


class SummaryAgent:
    """
    Parallel agent for generating multi-level meeting summaries.
    
    Uses Gemini 2.5 Flash to produce:
    - One-line caption
    - 3-sentence executive summary
    - Detailed bullet points
    - Key decisions
    """
    
    SUMMARY_PROMPT = """You are an expert meeting summarizer. Analyze the following meeting transcript and generate a comprehensive summary.

{context_section}

TRANSCRIPT:
{transcript}

Generate the following outputs in JSON format:
{{
    "one_liner": "A concise one-line summary (max 140 characters)",
    "executive_summary": "A 3-sentence executive summary covering the main points",
    "key_points": [
        "Bullet point 1",
        "Bullet point 2",
        "... (5-12 bullet points covering key discussion points)"
    ],
    "decisions": [
        "Decision 1 that was made",
        "Decision 2 that was made"
    ],
    "key_topics": [
        "Topic 1",
        "Topic 2"
    ]
}}

Guidelines:
- Be concise but comprehensive
- Focus on actionable insights
- Highlight decisions and agreements
- Identify main topics discussed
- Use professional language

Respond ONLY with the JSON object, no additional text."""

    def __init__(self, model: genai.GenerativeModel):
        self.model = model
        logger.info("SummaryAgent initialized")
    
    @trace_span("summary_agent.generate_summary")
    def generate_summary(
        self,
        transcript: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate multi-level summary from transcript.
        
        Args:
            transcript: The meeting transcript
            context: Optional context from memory
            
        Returns:
            Dictionary containing all summary levels
        """
        logger.debug("Generating summary", extra={"transcript_length": len(transcript)})
        
        # Build context section
        context_section = ""
        if context and context.get("relevant_meetings"):
            context_section = "RELEVANT CONTEXT FROM PAST MEETINGS:\n"
            for meeting in context["relevant_meetings"][:3]:
                context_section += f"- {meeting.get('summary', '')}\n"
            context_section += "\n"
        
        # Generate prompt
        prompt = self.SUMMARY_PROMPT.format(
            context_section=context_section,
            transcript=self._truncate_transcript(transcript),
        )
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_response(response.text)
            
            logger.info("Summary generated successfully", extra={
                "key_points_count": len(result.get("key_points", [])),
                "decisions_count": len(result.get("decisions", [])),
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return self._fallback_summary(transcript)
    
    def _truncate_transcript(self, transcript: str, max_chars: int = 30000) -> str:
        """Truncate transcript if too long, keeping beginning and end."""
        if len(transcript) <= max_chars:
            return transcript
        
        half = max_chars // 2
        return f"{transcript[:half]}\n\n[... transcript truncated ...]\n\n{transcript[-half:]}"
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON response from the model."""
        # Clean up response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r"```json?\n?", "", text)
            text = re.sub(r"```\n?$", "", text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, extracting manually")
            return self._extract_summary_manually(response_text)
    
    def _extract_summary_manually(self, text: str) -> Dict[str, Any]:
        """Fallback manual extraction if JSON parsing fails."""
        return {
            "one_liner": text[:140] if len(text) > 140 else text,
            "executive_summary": text[:500],
            "key_points": [],
            "decisions": [],
            "key_topics": [],
        }
    
    def _fallback_summary(self, transcript: str) -> Dict[str, Any]:
        """Generate a basic fallback summary if LLM fails."""
        words = transcript.split()
        return {
            "one_liner": f"Meeting discussion ({len(words)} words)",
            "executive_summary": "Meeting summary could not be generated. Please review the transcript manually.",
            "key_points": ["Review transcript for details"],
            "decisions": [],
            "key_topics": [],
        }
