"""
ContextCompactor - Context compaction for efficient memory usage.

Compresses long meeting transcripts and memories into compact representations.
"""
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from config import config
from observability.logger import get_logger


logger = get_logger(__name__)


class ContextCompactor:
    """
    Compacts long contexts into shorter, meaningful representations.
    
    Uses LLM to:
    - Summarize long transcripts
    - Extract key information
    - Create compact memory entries
    """
    
    COMPACTION_PROMPT = """You are a context compaction specialist. Your task is to compress 
the following content into a compact representation that preserves the most important information.

CONTENT TO COMPACT:
{content}

Create a compact version (max {max_tokens} tokens) that includes:
1. Key decisions made
2. Important action items
3. Critical topics discussed
4. Any unresolved issues
5. Names of key participants

Output format:
DECISIONS: [list key decisions]
ACTIONS: [list action items with owners]
TOPICS: [list main topics]
PARTICIPANTS: [list key people]
SUMMARY: [2-3 sentence summary]

Be concise but comprehensive."""

    def __init__(self):
        genai.configure(api_key=config.gemini.api_key)
        self.model = genai.GenerativeModel(config.gemini.model)
        logger.info("ContextCompactor initialized")
    
    def compact(
        self,
        content: str,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """
        Compact content into a shorter representation.
        
        Args:
            content: The content to compact
            max_tokens: Maximum tokens in output
            
        Returns:
            Compacted content with structured sections
        """
        logger.debug(f"Compacting content of length: {len(content)}")
        
        # If content is already short, return as-is
        if len(content.split()) < max_tokens * 2:
            return {
                "original_length": len(content),
                "compacted": content,
                "was_compacted": False,
            }
        
        try:
            prompt = self.COMPACTION_PROMPT.format(
                content=content[:30000],  # Limit input
                max_tokens=max_tokens,
            )
            
            response = self.model.generate_content(prompt)
            compacted = response.text.strip()
            
            # Parse structured output
            parsed = self._parse_compacted(compacted)
            
            logger.info(f"Content compacted: {len(content)} -> {len(compacted)} chars")
            
            return {
                "original_length": len(content),
                "compacted_length": len(compacted),
                "compacted": compacted,
                "parsed": parsed,
                "was_compacted": True,
            }
            
        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            # Fallback: simple truncation
            return {
                "original_length": len(content),
                "compacted": content[:max_tokens * 4],  # Rough approximation
                "was_compacted": False,
                "error": str(e),
            }
    
    def _parse_compacted(self, text: str) -> Dict[str, List[str]]:
        """Parse structured compacted output."""
        sections = {
            "decisions": [],
            "actions": [],
            "topics": [],
            "participants": [],
            "summary": "",
        }
        
        current_section = None
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            lower_line = line.lower()
            if lower_line.startswith("decisions:"):
                current_section = "decisions"
                content = line.split(":", 1)[1].strip()
                if content:
                    sections["decisions"].append(content)
            elif lower_line.startswith("actions:"):
                current_section = "actions"
                content = line.split(":", 1)[1].strip()
                if content:
                    sections["actions"].append(content)
            elif lower_line.startswith("topics:"):
                current_section = "topics"
                content = line.split(":", 1)[1].strip()
                if content:
                    sections["topics"].append(content)
            elif lower_line.startswith("participants:"):
                current_section = "participants"
                content = line.split(":", 1)[1].strip()
                if content:
                    sections["participants"].append(content)
            elif lower_line.startswith("summary:"):
                current_section = "summary"
                sections["summary"] = line.split(":", 1)[1].strip()
            elif current_section and current_section != "summary":
                # Add to current section
                if line.startswith("-") or line.startswith("•"):
                    sections[current_section].append(line[1:].strip())
                else:
                    sections[current_section].append(line)
            elif current_section == "summary":
                sections["summary"] += " " + line
        
        return sections
    
    def compact_multiple(
        self,
        contents: List[str],
        max_tokens_per_item: int = 200,
    ) -> str:
        """
        Compact multiple content pieces into a single context.
        
        Args:
            contents: List of content strings
            max_tokens_per_item: Max tokens per item
            
        Returns:
            Single compacted string
        """
        compacted_items = []
        
        for i, content in enumerate(contents):
            result = self.compact(content, max_tokens_per_item)
            compacted_items.append(f"[Item {i+1}] {result['compacted']}")
        
        return "\n\n".join(compacted_items)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: ~4 characters per token for English
        return len(text) // 4
    
    def truncate_to_context_window(
        self,
        text: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Truncate text to fit within context window.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (defaults to config)
            
        Returns:
            Truncated text
        """
        max_tokens = max_tokens or config.memory.context_window_size
        estimated = self.estimate_tokens(text)
        
        if estimated <= max_tokens:
            return text
        
        # Truncate with some buffer
        char_limit = max_tokens * 4  # Approximate
        return text[:char_limit] + "\n[... truncated ...]"
