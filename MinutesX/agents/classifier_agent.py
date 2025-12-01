"""
ClassifierAgent - Meeting type classification agent.

Classifies meetings into business categories:
- Sales
- Product
- Engineering
- Marketing
- Legal
- HR
- Finance
- General
"""
import json
import re
from typing import Any, Dict, List

import google.generativeai as genai

from observability.logger import get_logger
from observability.tracer import trace_span


logger = get_logger(__name__)


# Meeting categories with descriptions
MEETING_CATEGORIES = {
    "sales": "Sales calls, client meetings, deal discussions, pipeline reviews",
    "product": "Product planning, roadmap discussions, feature prioritization",
    "engineering": "Technical discussions, architecture reviews, sprint planning",
    "marketing": "Campaign planning, brand discussions, market analysis",
    "legal": "Contract reviews, compliance discussions, legal matters",
    "hr": "Hiring, performance reviews, team management, HR policies",
    "finance": "Budget reviews, financial planning, expense discussions",
    "operations": "Process improvements, logistics, operational planning",
    "strategy": "Strategic planning, company vision, long-term goals",
    "customer_success": "Customer support, account management, satisfaction reviews",
    "general": "Team meetings, all-hands, miscellaneous discussions",
}


class ClassifierAgent:
    """
    Parallel agent for classifying meeting types.
    
    Uses NLU to categorize meetings and extract relevant metadata.
    """
    
    CLASSIFIER_PROMPT = """You are an expert at classifying business meetings into categories.

MEETING TRANSCRIPT:
{transcript}

Analyze this meeting and classify it into the most appropriate category.

Available categories:
{categories}

Return a JSON object:
{{
    "primary_category": "category_name",
    "secondary_category": "category_name or null",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of classification",
    "keywords": ["key", "terms", "that", "influenced", "classification"],
    "sentiment": "positive|neutral|negative",
    "urgency_level": "high|medium|low",
    "meeting_type": "recurring|one-off|follow-up|kickoff|review"
}}

Consider:
- Main topics discussed
- Participants' roles (if identifiable)
- Action items and decisions
- Industry-specific terminology

Respond ONLY with the JSON object."""

    def __init__(self, model: genai.GenerativeModel):
        self.model = model
        self.categories = MEETING_CATEGORIES
        logger.info("ClassifierAgent initialized")
    
    @trace_span("classifier_agent.classify_meeting")
    def classify_meeting(self, transcript: str) -> Dict[str, Any]:
        """
        Classify the meeting into business categories.
        
        Args:
            transcript: The meeting transcript
            
        Returns:
            Classification result with category, confidence, and metadata
        """
        logger.debug("Classifying meeting", extra={"transcript_length": len(transcript)})
        
        categories_str = "\n".join(
            f"- {cat}: {desc}" for cat, desc in self.categories.items()
        )
        
        prompt = self.CLASSIFIER_PROMPT.format(
            transcript=self._sample_transcript(transcript),
            categories=categories_str,
        )
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_response(response.text)
            
            # Validate category
            if result.get("primary_category") not in self.categories:
                result["primary_category"] = "general"
            
            logger.info("Meeting classified", extra={
                "category": result.get("primary_category"),
                "confidence": result.get("confidence"),
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._fallback_classification()
    
    def _sample_transcript(self, transcript: str, max_chars: int = 15000) -> str:
        """Sample representative parts of transcript for classification."""
        if len(transcript) <= max_chars:
            return transcript
        
        # Take samples from different parts
        chunk_size = max_chars // 3
        return (
            f"{transcript[:chunk_size]}\n\n"
            f"[... middle portion ...]\n\n"
            f"{transcript[-chunk_size:]}"
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
            logger.warning("Failed to parse classification JSON")
            return self._fallback_classification()
    
    def _fallback_classification(self) -> Dict[str, Any]:
        """Return fallback classification if LLM fails."""
        return {
            "primary_category": "general",
            "secondary_category": None,
            "confidence": 0.5,
            "reasoning": "Classification could not be determined",
            "keywords": [],
            "sentiment": "neutral",
            "urgency_level": "medium",
            "meeting_type": "one-off",
        }
    
    def get_category_icon(self, category: str) -> str:
        """Get emoji icon for category."""
        icons = {
            "sales": "💰",
            "product": "📦",
            "engineering": "⚙️",
            "marketing": "📢",
            "legal": "⚖️",
            "hr": "👥",
            "finance": "📊",
            "operations": "🔧",
            "strategy": "🎯",
            "customer_success": "🤝",
            "general": "📝",
        }
        return icons.get(category, "📋")
