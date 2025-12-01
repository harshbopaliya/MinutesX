"""
ReviewerAgent - Quality review and merge agent.

Sequential agent that:
- Merges outputs from parallel agents
- Removes duplicates
- Performs quality checks
- Refines final output
"""
import json
import re
from typing import Any, Dict, List

import google.generativeai as genai

from observability.logger import get_logger
from observability.tracer import trace_span


logger = get_logger(__name__)


class ReviewerAgent:
    """
    Sequential agent for reviewing and merging outputs from parallel agents.
    
    Runs after all parallel agents complete to:
    - Merge and deduplicate results
    - Ensure consistency
    - Quality check outputs
    - Create final polished result
    """
    
    REVIEW_PROMPT = """You are a quality reviewer for meeting notes. Your job is to review and improve the outputs 
from multiple AI agents that analyzed a meeting.

ORIGINAL TRANSCRIPT EXCERPT:
{transcript_excerpt}

AGENT OUTPUTS:

SUMMARY AGENT OUTPUT:
{summary_output}

ACTION ITEMS AGENT OUTPUT:
{action_output}

CAPTION AGENT OUTPUT:
{caption_output}

CLASSIFICATION AGENT OUTPUT:
{classifier_output}

Your tasks:
1. Review all outputs for accuracy and consistency
2. Remove any duplicate action items
3. Ensure the summary aligns with action items
4. Verify the classification makes sense
5. Select the best caption
6. Add any missing important points

Return a refined, merged result as JSON:
{{
    "caption": "The best one-liner caption",
    "summary": {{
        "one_liner": "Refined one-liner",
        "executive_summary": "Refined 3-sentence summary",
        "key_points": ["Deduplicated and refined key points"],
        "decisions": ["Key decisions made"]
    }},
    "action_items": [
        {{
            "id": 1,
            "description": "Refined action item",
            "owner": "Owner Name",
            "due_date": "YYYY-MM-DD or null",
            "priority": "high|medium|low"
        }}
    ],
    "category": "primary_category",
    "key_topics": ["topic1", "topic2"],
    "confidence": 0.0 to 1.0,
    "quality_notes": "Any quality issues found or improvements made"
}}

Respond ONLY with the JSON object."""

    def __init__(self, model: genai.GenerativeModel):
        self.model = model
        logger.info("ReviewerAgent initialized")
    
    @trace_span("reviewer_agent.review_and_merge")
    def review_and_merge(
        self,
        parallel_results: Dict[str, Any],
        transcript: str,
        meeting_id: str,
    ) -> Dict[str, Any]:
        """
        Review and merge outputs from parallel agents.
        
        Args:
            parallel_results: Dictionary of outputs from parallel agents
            transcript: Original transcript for reference
            meeting_id: Meeting identifier
            
        Returns:
            Merged and refined result
        """
        logger.debug("Reviewing and merging results", extra={
            "meeting_id": meeting_id,
            "agent_count": len(parallel_results),
        })
        
        # Check for errors in parallel results
        errors = [k for k, v in parallel_results.items() if "error" in v]
        if errors:
            logger.warning(f"Some agents had errors: {errors}")
        
        # Build review prompt
        prompt = self.REVIEW_PROMPT.format(
            transcript_excerpt=transcript[:3000],
            summary_output=json.dumps(parallel_results.get("summary", {}), indent=2),
            action_output=json.dumps(parallel_results.get("action", {}), indent=2),
            caption_output=json.dumps(parallel_results.get("caption", {}), indent=2),
            classifier_output=json.dumps(parallel_results.get("classifier", {}), indent=2),
        )
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_response(response.text)
            
            # Ensure required fields exist
            result = self._ensure_required_fields(result, parallel_results)
            
            logger.info("Review completed", extra={
                "meeting_id": meeting_id,
                "action_items_count": len(result.get("action_items", [])),
                "confidence": result.get("confidence"),
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Review failed: {e}")
            return self._fallback_merge(parallel_results)
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from model."""
        text = response_text.strip()
        
        if text.startswith("```"):
            text = re.sub(r"```json?\n?", "", text)
            text = re.sub(r"```\n?$", "", text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse reviewer JSON")
            return {}
    
    def _ensure_required_fields(
        self,
        result: Dict[str, Any],
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ensure all required fields are present."""
        # Default values
        defaults = {
            "caption": "",
            "summary": {"one_liner": "", "executive_summary": "", "key_points": [], "decisions": []},
            "action_items": [],
            "category": "general",
            "key_topics": [],
            "confidence": 0.8,
        }
        
        for key, default_value in defaults.items():
            if key not in result or not result[key]:
                # Try to get from fallback
                if key == "caption":
                    result[key] = fallback.get("caption", {}).get("one_liner", default_value)
                elif key == "summary":
                    result[key] = fallback.get("summary", default_value)
                elif key == "action_items":
                    result[key] = fallback.get("action", {}).get("action_items", default_value)
                elif key == "category":
                    result[key] = fallback.get("classifier", {}).get("primary_category", default_value)
                else:
                    result[key] = default_value
        
        return result
    
    def _fallback_merge(self, parallel_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic merge if LLM review fails."""
        summary_result = parallel_results.get("summary", {})
        action_result = parallel_results.get("action", {})
        caption_result = parallel_results.get("caption", {})
        classifier_result = parallel_results.get("classifier", {})
        
        return {
            "caption": caption_result.get("one_liner", "Meeting summary"),
            "summary": {
                "one_liner": summary_result.get("one_liner", ""),
                "executive_summary": summary_result.get("executive_summary", ""),
                "key_points": summary_result.get("key_points", []),
                "decisions": summary_result.get("decisions", []),
            },
            "action_items": action_result.get("action_items", []),
            "category": classifier_result.get("primary_category", "general"),
            "key_topics": summary_result.get("key_topics", []),
            "confidence": 0.7,
            "quality_notes": "Fallback merge used - LLM review failed",
        }
    
    def deduplicate_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate action items based on description similarity."""
        seen_descriptions = set()
        unique_actions = []
        
        for action in actions:
            desc = action.get("description", "").lower().strip()
            # Simple deduplication - could use embeddings for better matching
            desc_key = " ".join(desc.split()[:5])  # First 5 words as key
            
            if desc_key not in seen_descriptions:
                seen_descriptions.add(desc_key)
                unique_actions.append(action)
        
        return unique_actions
