"""
ActionItemAgent - Action item extraction and assignment agent.

Extracts:
- Action items from meeting transcript
- Suggested owners (matched to participants)
- Due dates (if mentioned)
- Priority levels
"""
import json
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from observability.logger import get_logger
from observability.tracer import trace_span


logger = get_logger(__name__)


class ActionItemAgent:
    """
    Parallel agent for extracting action items from meeting transcripts.
    
    Features:
    - Identifies action items and tasks
    - Matches owners to participants
    - Extracts due dates
    - Assigns priority levels
    """
    
    ACTION_PROMPT = """You are an expert at identifying action items from meeting transcripts.

MEETING TRANSCRIPT:
{transcript}

PARTICIPANTS (if known):
{participants}

Extract all action items from this meeting. For each action item, identify:
1. The task description
2. Who is responsible (owner) - match to participants if possible
3. Due date if mentioned
4. Priority (high/medium/low) based on urgency indicators

Return a JSON array of action items:
{{
    "action_items": [
        {{
            "id": 1,
            "description": "Clear description of what needs to be done",
            "owner": "Person Name or 'Unassigned'",
            "due_date": "YYYY-MM-DD or null if not specified",
            "priority": "high|medium|low",
            "context": "Brief context from the meeting"
        }}
    ],
    "follow_ups": [
        "Follow-up items that aren't specific action items"
    ]
}}

Look for phrases like:
- "I'll do...", "I will...", "Let me..."
- "Can you...", "Could you...", "Please..."
- "We need to...", "We should..."
- "Action item:", "TODO:", "Next steps:"
- "By [date]...", "Before [event]..."

Respond ONLY with the JSON object."""

    def __init__(self, model: genai.GenerativeModel):
        self.model = model
        logger.info("ActionItemAgent initialized")
    
    @trace_span("action_agent.extract_actions")
    def extract_actions(
        self,
        transcript: str,
        participants: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract action items from transcript.
        
        Args:
            transcript: The meeting transcript
            participants: Optional list of participant names
            
        Returns:
            Dictionary with action_items and follow_ups
        """
        logger.debug("Extracting action items", extra={
            "transcript_length": len(transcript),
            "participant_count": len(participants) if participants else 0,
        })
        
        participants_str = ", ".join(participants) if participants else "Unknown"
        
        prompt = self.ACTION_PROMPT.format(
            transcript=self._truncate_transcript(transcript),
            participants=participants_str,
        )
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_response(response.text)
            
            # Post-process: validate and normalize owners
            if participants:
                result["action_items"] = self._resolve_owners(
                    result.get("action_items", []),
                    participants
                )
            
            logger.info("Action items extracted", extra={
                "action_count": len(result.get("action_items", [])),
                "follow_up_count": len(result.get("follow_ups", [])),
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Action extraction failed: {e}")
            return {"action_items": [], "follow_ups": []}
    
    def _truncate_transcript(self, transcript: str, max_chars: int = 25000) -> str:
        """Truncate transcript if too long."""
        if len(transcript) <= max_chars:
            return transcript
        return transcript[:max_chars] + "\n[... truncated ...]"
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from model."""
        text = response_text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            text = re.sub(r"```json?\n?", "", text)
            text = re.sub(r"```\n?$", "", text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse action items JSON")
            return {"action_items": [], "follow_ups": []}
    
    def _resolve_owners(
        self,
        action_items: List[Dict[str, Any]],
        participants: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Resolve owner names to match participant list.
        
        Uses fuzzy matching to find best participant match.
        """
        participant_lower = {p.lower(): p for p in participants}
        
        for item in action_items:
            owner = item.get("owner", "").lower()
            
            if owner in participant_lower:
                item["owner"] = participant_lower[owner]
            else:
                # Try partial match
                for p_lower, p_original in participant_lower.items():
                    if owner in p_lower or p_lower in owner:
                        item["owner"] = p_original
                        break
                else:
                    # Check for first name match
                    owner_first = owner.split()[0] if owner else ""
                    for p_lower, p_original in participant_lower.items():
                        if owner_first and owner_first in p_lower.split()[0]:
                            item["owner"] = p_original
                            break
        
        return action_items
    
    def format_for_export(
        self,
        action_items: List[Dict[str, Any]],
        format_type: str = "markdown",
    ) -> str:
        """
        Format action items for export.
        
        Args:
            action_items: List of action items
            format_type: Output format (markdown, plain, jira)
        """
        if format_type == "markdown":
            lines = ["## Action Items\n"]
            for item in action_items:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    item.get("priority", "medium"), "⚪"
                )
                due = f" (Due: {item['due_date']})" if item.get("due_date") else ""
                lines.append(
                    f"- {priority_emoji} **{item['description']}**\n"
                    f"  - Owner: {item.get('owner', 'Unassigned')}{due}\n"
                )
            return "\n".join(lines)
        
        elif format_type == "plain":
            lines = ["ACTION ITEMS:"]
            for i, item in enumerate(action_items, 1):
                due = f" (Due: {item['due_date']})" if item.get("due_date") else ""
                lines.append(
                    f"{i}. {item['description']} - {item.get('owner', 'Unassigned')}{due}"
                )
            return "\n".join(lines)
        
        return str(action_items)
