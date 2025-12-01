"""
MemoryAgent - Long-term memory management agent.

Handles:
- Context retrieval from past meetings
- Meeting storage to memory bank
- Context compaction for old meetings
- Semantic search across meeting history
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from memory.memory_bank import MemoryBank
from observability.logger import get_logger
from observability.tracer import trace_span


logger = get_logger(__name__)


class MemoryAgent:
    """
    Sequential agent for memory management and context retrieval.
    
    Features:
    - Semantic search across past meetings
    - Context compaction for efficiency
    - Long-term knowledge persistence
    """
    
    COMPACTION_PROMPT = """You are a context compaction expert. Compress the following meeting information 
into a concise summary that preserves the most important facts, decisions, and action items.

MEETING DATA:
{meeting_data}

Create a compact summary (max 200 words) that includes:
- Key decisions made
- Important action items and their status
- Critical topics discussed
- Any unresolved issues

Return ONLY the compact summary text, no JSON or formatting."""

    def __init__(self, model: genai.GenerativeModel):
        self.model = model
        self.memory_bank = MemoryBank()
        logger.info("MemoryAgent initialized")
    
    @trace_span("memory_agent.retrieve_context")
    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from past meetings.
        
        Args:
            query: Search query (usually beginning of transcript)
            limit: Maximum number of results
            
        Returns:
            Dictionary with relevant_meetings and summary
        """
        logger.debug("Retrieving context", extra={"query_length": len(query)})
        
        try:
            # Search memory bank
            results = self.memory_bank.search(query, limit=limit)
            
            context = {
                "relevant_meetings": results,
                "meeting_count": len(results),
                "has_context": len(results) > 0,
            }
            
            # Generate context summary if we have results
            if results:
                context["context_summary"] = self._summarize_context(results)
            
            logger.info("Context retrieved", extra={
                "results_count": len(results),
            })
            
            return context
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return {"relevant_meetings": [], "meeting_count": 0, "has_context": False}
    
    @trace_span("memory_agent.store_meeting")
    def store_meeting(
        self,
        meeting_id: str,
        meeting_data: Dict[str, Any],
    ) -> bool:
        """
        Store meeting data to long-term memory.
        
        Args:
            meeting_id: Unique meeting identifier
            meeting_data: Processed meeting data to store
            
        Returns:
            Success status
        """
        logger.debug("Storing meeting", extra={"meeting_id": meeting_id})
        
        try:
            # Prepare document for storage
            document = {
                "meeting_id": meeting_id,
                "timestamp": datetime.utcnow().isoformat(),
                "caption": meeting_data.get("caption", ""),
                "summary": meeting_data.get("summary", {}),
                "action_items": meeting_data.get("action_items", []),
                "category": meeting_data.get("category", "general"),
                "decisions": meeting_data.get("decisions", []),
                "key_topics": meeting_data.get("key_topics", []),
            }
            
            # Create searchable text
            searchable_text = self._create_searchable_text(document)
            
            # Store in memory bank
            self.memory_bank.store(
                doc_id=meeting_id,
                text=searchable_text,
                metadata=document,
            )
            
            logger.info("Meeting stored", extra={"meeting_id": meeting_id})
            return True
            
        except Exception as e:
            logger.error(f"Meeting storage failed: {e}", extra={"meeting_id": meeting_id})
            return False
    
    @trace_span("memory_agent.compact_old_memories")
    def compact_old_memories(self, days_threshold: int = 90) -> int:
        """
        Compact old meeting memories to save space.
        
        Summarizes meetings older than threshold into compact form.
        
        Args:
            days_threshold: Age in days after which to compact
            
        Returns:
            Number of meetings compacted
        """
        logger.info(f"Starting memory compaction for meetings older than {days_threshold} days")
        
        try:
            # Get old meetings
            old_meetings = self.memory_bank.get_old_documents(days_threshold)
            compacted_count = 0
            
            for meeting in old_meetings:
                compact_text = self._compact_meeting(meeting)
                self.memory_bank.update_document(
                    doc_id=meeting["meeting_id"],
                    text=compact_text,
                    metadata={"compacted": True, **meeting},
                )
                compacted_count += 1
            
            logger.info(f"Compacted {compacted_count} old meetings")
            return compacted_count
            
        except Exception as e:
            logger.error(f"Memory compaction failed: {e}")
            return 0
    
    def _summarize_context(self, meetings: List[Dict[str, Any]]) -> str:
        """Create a brief summary of relevant context."""
        summaries = []
        for meeting in meetings[:3]:
            if meeting.get("caption"):
                summaries.append(f"- {meeting['caption']}")
        return "\n".join(summaries) if summaries else "No relevant context found."
    
    def _create_searchable_text(self, document: Dict[str, Any]) -> str:
        """Create searchable text from meeting document."""
        parts = [
            document.get("caption", ""),
            document.get("summary", {}).get("executive_summary", ""),
            " ".join(document.get("key_topics", [])),
            " ".join(document.get("decisions", [])),
        ]
        
        # Add action item descriptions
        for action in document.get("action_items", []):
            parts.append(action.get("description", ""))
        
        return " ".join(filter(None, parts))
    
    def _compact_meeting(self, meeting: Dict[str, Any]) -> str:
        """Use LLM to compact a meeting into a brief summary."""
        try:
            prompt = self.COMPACTION_PROMPT.format(
                meeting_data=json.dumps(meeting, indent=2)[:5000]
            )
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Meeting compaction LLM call failed: {e}")
            return meeting.get("caption", "Meeting summary unavailable")
    
    def search_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search meetings by topic."""
        return self.memory_bank.search(topic, limit=limit)
    
    def get_meeting(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific meeting by ID."""
        return self.memory_bank.get_document(meeting_id)
