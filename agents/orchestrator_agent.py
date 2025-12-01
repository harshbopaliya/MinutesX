"""
MeetingOrchestratorAgent - Main orchestration agent using Google ADK.

This is the root agent that coordinates all sub-agents for meeting processing.
Uses Gemini 2.5 Flash as the primary LLM.

Features demonstrated:
- Multi-agent system (orchestrator pattern)
- Parallel agent execution
- Sequential agent execution
- Session management
- A2A Protocol integration
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import uuid

# Google ADK imports
try:
    from google import adk
    from google.adk import Agent, Tool
    from google.adk.sessions import InMemorySessionService
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("Warning: Google ADK not installed. Using mock implementation.")

import google.generativeai as genai

from config import config, Config
from observability.logger import get_logger, log_agent_execution
from observability.metrics import metrics_collector
from observability.tracer import trace_span


logger = get_logger(__name__)


@dataclass
class MeetingResult:
    """Structured output from meeting processing."""
    meeting_id: str
    caption: str
    summary: Dict[str, Any]
    action_items: List[Dict[str, Any]]
    category: str
    decisions: List[str]
    key_topics: List[str]
    participants: List[str]
    confidence: float
    processed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "meeting_id": self.meeting_id,
            "caption": self.caption,
            "summary": self.summary,
            "action_items": self.action_items,
            "category": self.category,
            "decisions": self.decisions,
            "key_topics": self.key_topics,
            "participants": self.participants,
            "confidence": self.confidence,
            "processed_at": self.processed_at.isoformat(),
        }


class MeetingOrchestratorAgent:
    """
    Main orchestrator agent for MinutesX.
    
    Coordinates parallel and sequential agents to process meeting transcripts
    and generate comprehensive meeting notes.
    
    Architecture:
    - Parallel: SummaryAgent, ActionItemAgent, CaptionAgent, ClassifierAgent
    - Sequential: MemoryAgent (context retrieval), ReviewerAgent (final merge)
    """
    
    ORCHESTRATOR_PROMPT = """You are the Meeting Orchestrator for MinutesX, an intelligent meeting notes system.

Your role is to coordinate the processing of meeting transcripts and ensure high-quality outputs.

You have access to the following sub-agents:
1. SummaryAgent - Generates multi-level summaries
2. ActionItemAgent - Extracts action items with owners and due dates
3. CaptionAgent - Creates shareable one-line captions
4. ClassifierAgent - Categorizes the meeting type
5. MemoryAgent - Retrieves relevant context from past meetings
6. ReviewerAgent - Merges and refines all outputs

For each meeting, you will:
1. Receive the transcript
2. Dispatch to parallel agents simultaneously
3. Collect and merge results
4. Send to ReviewerAgent for final quality check
5. Store results in memory for future reference

Always maintain context awareness and ensure action items are properly assigned."""

    def __init__(
        self,
        session_service: Optional[Any] = None,
        cfg: Config = config,
    ):
        self.config = cfg
        self.session_service = session_service
        
        # Initialize Gemini
        genai.configure(api_key=self.config.gemini.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.config.gemini.model,
            generation_config={
                "temperature": self.config.gemini.temperature,
                "max_output_tokens": self.config.gemini.max_tokens,
            }
        )
        
        # Import agents (lazy to avoid circular imports)
        from agents.summary_agent import SummaryAgent
        from agents.action_agent import ActionItemAgent
        from agents.caption_agent import CaptionAgent
        from agents.classifier_agent import ClassifierAgent
        from agents.memory_agent import MemoryAgent
        from agents.reviewer_agent import ReviewerAgent
        
        # Initialize sub-agents
        self.agents = {
            "summary": SummaryAgent(self.model),
            "action": ActionItemAgent(self.model),
            "caption": CaptionAgent(self.model),
            "classifier": ClassifierAgent(self.model),
            "memory": MemoryAgent(self.model),
            "reviewer": ReviewerAgent(self.model),
        }
        
        # A2A message handlers
        self.message_handlers: Dict[str, Callable] = {}
        
        logger.info("MeetingOrchestratorAgent initialized", extra={
            "model": self.config.gemini.model,
            "agents": list(self.agents.keys()),
        })
    
    @trace_span("orchestrator.process_meeting")
    def process_meeting(
        self,
        meeting_id: str,
        transcript: str,
        participants: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MeetingResult:
        """
        Process a meeting transcript through the multi-agent pipeline.
        
        Args:
            meeting_id: Unique identifier for the meeting
            transcript: The meeting transcript text
            participants: Optional list of participant names
            metadata: Optional additional metadata
            
        Returns:
            MeetingResult with all processed outputs
        """
        start_time = datetime.utcnow()
        logger.info(f"Processing meeting", extra={
            "meeting_id": meeting_id,
            "transcript_length": len(transcript),
        })
        
        try:
            # Step 1: Retrieve relevant context from memory (sequential)
            memory_context = self.agents["memory"].retrieve_context(transcript[:1000])
            
            # Step 2: Run parallel agents
            parallel_results = self._run_parallel_agents(
                transcript=transcript,
                participants=participants or [],
                context=memory_context,
            )
            
            # Step 3: Run reviewer agent (sequential)
            reviewed_result = self.agents["reviewer"].review_and_merge(
                parallel_results,
                transcript=transcript,
                meeting_id=meeting_id,
            )
            
            # Step 4: Store to memory
            self.agents["memory"].store_meeting(meeting_id, reviewed_result)
            
            # Step 5: Create final result
            result = MeetingResult(
                meeting_id=meeting_id,
                caption=reviewed_result.get("caption", ""),
                summary=reviewed_result.get("summary", {}),
                action_items=reviewed_result.get("action_items", []),
                category=reviewed_result.get("category", "General"),
                decisions=reviewed_result.get("decisions", []),
                key_topics=reviewed_result.get("key_topics", []),
                participants=participants or reviewed_result.get("participants", []),
                confidence=reviewed_result.get("confidence", 0.85),
            )
            
            # Record metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics_collector.record_meeting_processed(meeting_id, duration, "success")
            
            logger.info("Meeting processed successfully", extra={
                "meeting_id": meeting_id,
                "duration_seconds": duration,
                "action_items_count": len(result.action_items),
                "category": result.category,
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Meeting processing failed", extra={
                "meeting_id": meeting_id,
                "error": str(e),
            })
            metrics_collector.record_meeting_processed(meeting_id, 0, "failure")
            raise
    
    def _run_parallel_agents(
        self,
        transcript: str,
        participants: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute parallel agents concurrently.
        
        Runs these agents in parallel:
        - SummaryAgent
        - ActionItemAgent
        - CaptionAgent
        - ClassifierAgent
        """
        results = {}
        
        agent_tasks = [
            ("summary", lambda: self.agents["summary"].generate_summary(transcript, context)),
            ("action", lambda: self.agents["action"].extract_actions(transcript, participants)),
            ("caption", lambda: self.agents["caption"].generate_caption(transcript)),
            ("classifier", lambda: self.agents["classifier"].classify_meeting(transcript)),
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(task): name
                for name, task in agent_tasks
            }
            
            for future in as_completed(futures, timeout=self.config.agent.parallel_timeout):
                agent_name = futures[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                    log_agent_execution(agent_name, "success")
                    logger.debug(f"Agent {agent_name} completed", extra={"agent": agent_name})
                except Exception as e:
                    logger.error(f"Agent {agent_name} failed", extra={
                        "agent": agent_name,
                        "error": str(e),
                    })
                    results[agent_name] = {"error": str(e)}
                    log_agent_execution(agent_name, "failure")
        
        return results
    
    async def process_meeting_async(
        self,
        meeting_id: str,
        transcript: str,
        participants: Optional[List[str]] = None,
    ) -> MeetingResult:
        """Async version of process_meeting."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_meeting,
            meeting_id,
            transcript,
            participants,
        )
    
    def register_a2a_handler(self, agent_id: str, handler: Callable):
        """Register an A2A message handler for an agent."""
        self.message_handlers[agent_id] = handler
    
    def send_a2a_message(self, target: str, message: Dict[str, Any]):
        """Send an A2A message to a target agent."""
        if target in self.message_handlers:
            self.message_handlers[target](message)


def create_orchestrator(cfg: Config = config) -> MeetingOrchestratorAgent:
    """Factory function to create an orchestrator instance."""
    return MeetingOrchestratorAgent(cfg=cfg)
