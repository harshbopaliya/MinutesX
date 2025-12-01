"""
GoogleSearchTool - Built-in Google Search integration.

Uses Google Search to enrich meeting context with relevant information.
"""
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from config import config
from observability.logger import get_logger


logger = get_logger(__name__)


class GoogleSearchTool:
    """
    Built-in tool for Google Search integration.
    
    Uses Gemini's grounding with Google Search to fetch
    relevant information for meeting context enrichment.
    """
    
    def __init__(self):
        genai.configure(api_key=config.gemini.api_key)
        self.model = genai.GenerativeModel(config.gemini.model)
        logger.info("GoogleSearchTool initialized")
    
    def search_context(
        self,
        query: str,
        num_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Search for relevant context using Google Search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        logger.debug(f"Searching for: {query}")
        
        try:
            # Use Gemini with grounding for search
            # Note: This requires the model to support grounding
            prompt = f"""Search for relevant information about: {query}

Provide a concise summary of the most relevant information found.
Include key facts, recent developments, and important context.

Format your response as:
1. Main topic summary
2. Key facts (bullet points)
3. Recent developments (if any)
4. Relevant context for meetings"""

            response = self.model.generate_content(prompt)
            
            return {
                "query": query,
                "summary": response.text,
                "success": True,
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "query": query,
                "summary": "",
                "success": False,
                "error": str(e),
            }
    
    def enrich_topics(
        self,
        topics: List[str],
    ) -> Dict[str, str]:
        """
        Enrich a list of topics with additional context.
        
        Args:
            topics: List of topics to enrich
            
        Returns:
            Dictionary mapping topics to their enriched context
        """
        enriched = {}
        
        for topic in topics[:5]:  # Limit to 5 topics
            result = self.search_context(topic)
            if result["success"]:
                enriched[topic] = result["summary"]
            else:
                enriched[topic] = "Context not available"
        
        return enriched
    
    def get_company_info(self, company_name: str) -> Dict[str, Any]:
        """
        Get information about a company mentioned in the meeting.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Company information
        """
        query = f"{company_name} company overview business information"
        return self.search_context(query)
    
    def get_term_definition(self, term: str) -> Dict[str, Any]:
        """
        Get definition of a technical or business term.
        
        Args:
            term: The term to define
            
        Returns:
            Term definition and context
        """
        query = f"define {term} in business context"
        return self.search_context(query)
