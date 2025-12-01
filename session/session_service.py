"""
SessionService - Session and state management.

Provides:
- InMemorySessionService for ephemeral session state
- Session objects for tracking meeting processing
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from threading import Lock

from observability.logger import get_logger


logger = get_logger(__name__)


class SessionStatus(Enum):
    """Session status enumeration."""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class Session:
    """
    Represents a meeting processing session.
    
    Tracks:
    - Session lifecycle
    - Processing state
    - Agent results
    - Context data
    """
    session_id: str
    meeting_id: str
    transcript: str = ""
    status: SessionStatus = SessionStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Processing state
    participants: List[str] = field(default_factory=list)
    agent_results: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def set_status(self, status: SessionStatus):
        """Update session status."""
        self.status = status
        self.updated_at = datetime.utcnow()
    
    def add_agent_result(self, agent_name: str, result: Any):
        """Add result from an agent."""
        self.agent_results[agent_name] = {
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.updated_at = datetime.utcnow()
    
    def add_context(self, key: str, value: Any):
        """Add context data."""
        self.context[key] = value
        self.updated_at = datetime.utcnow()
    
    def add_error(self, error: str):
        """Record an error."""
        self.errors.append({
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "meeting_id": self.meeting_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "participants": self.participants,
            "agent_results": self.agent_results,
            "context": self.context,
            "metadata": self.metadata,
            "errors": self.errors,
            "transcript_length": len(self.transcript),
        }


class InMemorySessionService:
    """
    In-memory session service for managing meeting sessions.
    
    Features:
    - Thread-safe session management
    - Session lifecycle tracking
    - Pause/resume support for long-running operations
    """
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = Lock()
        logger.info("InMemorySessionService initialized")
    
    def create_session(
        self,
        meeting_id: str,
        transcript: str = "",
        participants: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            meeting_id: Meeting identifier
            transcript: Meeting transcript
            participants: List of participants
            metadata: Additional metadata
            
        Returns:
            New Session object
        """
        session_id = str(uuid.uuid4())
        
        session = Session(
            session_id=session_id,
            meeting_id=meeting_id,
            transcript=transcript,
            participants=participants or [],
            metadata=metadata or {},
        )
        
        with self._lock:
            self._sessions[session_id] = session
        
        logger.info(f"Session created: {session_id}", extra={
            "meeting_id": meeting_id,
            "session_id": session_id,
        })
        
        return session
    
    def start_session(
        self,
        meeting_id: str,
        transcript: str,
        **kwargs,
    ) -> Session:
        """
        Create and start a session (alias for create_session).
        
        Sets status to PROCESSING immediately.
        """
        session = self.create_session(meeting_id, transcript, **kwargs)
        session.set_status(SessionStatus.PROCESSING)
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def get_session_by_meeting(self, meeting_id: str) -> Optional[Session]:
        """Get the most recent session for a meeting."""
        sessions = [
            s for s in self._sessions.values()
            if s.meeting_id == meeting_id
        ]
        if sessions:
            return max(sessions, key=lambda s: s.created_at)
        return None
    
    def update_session(
        self,
        session_id: str,
        **updates,
    ) -> Optional[Session]:
        """Update session attributes."""
        session = self._sessions.get(session_id)
        if session:
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.updated_at = datetime.utcnow()
        return session
    
    def complete_session(self, session_id: str) -> Optional[Session]:
        """Mark a session as completed."""
        session = self._sessions.get(session_id)
        if session:
            session.set_status(SessionStatus.COMPLETED)
            logger.info(f"Session completed: {session_id}")
        return session
    
    def fail_session(self, session_id: str, error: str) -> Optional[Session]:
        """Mark a session as failed."""
        session = self._sessions.get(session_id)
        if session:
            session.add_error(error)
            session.set_status(SessionStatus.FAILED)
            logger.error(f"Session failed: {session_id}", extra={"error": error})
        return session
    
    def pause_session(self, session_id: str) -> Optional[Session]:
        """
        Pause a session for long-running operations.
        
        Supports the pause/resume pattern for agents.
        """
        session = self._sessions.get(session_id)
        if session and session.status == SessionStatus.PROCESSING:
            session.set_status(SessionStatus.PAUSED)
            logger.info(f"Session paused: {session_id}")
        return session
    
    def resume_session(self, session_id: str) -> Optional[Session]:
        """Resume a paused session."""
        session = self._sessions.get(session_id)
        if session and session.status == SessionStatus.PAUSED:
            session.set_status(SessionStatus.PROCESSING)
            logger.info(f"Session resumed: {session_id}")
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Session deleted: {session_id}")
                return True
        return False
    
    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        meeting_id: Optional[str] = None,
    ) -> List[Session]:
        """List sessions with optional filters."""
        sessions = list(self._sessions.values())
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        if meeting_id:
            sessions = [s for s in sessions if s.meeting_id == meeting_id]
        
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session service statistics."""
        status_counts = {}
        for session in self._sessions.values():
            status = session.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_sessions": len(self._sessions),
            "status_counts": status_counts,
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours."""
        cutoff = datetime.utcnow()
        removed = 0
        
        with self._lock:
            to_remove = []
            for session_id, session in self._sessions.items():
                age = (cutoff - session.created_at).total_seconds() / 3600
                if age > max_age_hours and session.status in [
                    SessionStatus.COMPLETED,
                    SessionStatus.FAILED,
                ]:
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                del self._sessions[session_id]
                removed += 1
        
        if removed:
            logger.info(f"Cleaned up {removed} old sessions")
        
        return removed
