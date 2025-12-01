"""
A2A Protocol - Agent-to-Agent communication protocol.

Implements the A2A protocol for structured agent communication.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from observability.logger import get_logger


logger = get_logger(__name__)


class MessageType(Enum):
    """Types of A2A messages."""
    TASK_REQUEST = "TASK_REQUEST"
    TASK_RESULT = "TASK_RESULT"
    TASK_ACK = "TASK_ACK"
    TASK_NACK = "TASK_NACK"
    STATUS_UPDATE = "STATUS_UPDATE"
    ERROR = "ERROR"
    HEARTBEAT = "HEARTBEAT"


@dataclass
class Message:
    """
    A2A Protocol message.
    
    Represents a message passed between agents.
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.TASK_REQUEST
    source_agent: str = ""
    target_agent: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data.get("message_type", "TASK_REQUEST")),
            source_agent=data.get("source_agent", ""),
            target_agent=data.get("target_agent", ""),
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


class A2AProtocol:
    """
    A2A Protocol implementation.
    
    Provides structured communication between agents with:
    - Request/response patterns
    - Acknowledgments
    - Error handling
    - Correlation tracking
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._handlers: Dict[MessageType, Callable] = {}
        self._pending_requests: Dict[str, Message] = {}
        self._message_history: List[Message] = []
        
        logger.info(f"A2AProtocol initialized for agent: {agent_id}")
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Optional[Message]],
    ):
        """
        Register a handler for a message type.
        
        Args:
            message_type: Type of message to handle
            handler: Callback function that processes the message
        """
        self._handlers[message_type] = handler
        logger.debug(f"Handler registered for {message_type.value}")
    
    def create_request(
        self,
        target_agent: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Create a task request message.
        
        Args:
            target_agent: ID of the target agent
            payload: Request payload
            metadata: Optional metadata
            
        Returns:
            Request message
        """
        message = Message(
            message_type=MessageType.TASK_REQUEST,
            source_agent=self.agent_id,
            target_agent=target_agent,
            payload=payload,
            metadata=metadata or {},
        )
        
        self._pending_requests[message.message_id] = message
        self._message_history.append(message)
        
        logger.debug(f"Created request: {message.message_id} -> {target_agent}")
        return message
    
    def create_response(
        self,
        request: Message,
        result: Any,
        success: bool = True,
    ) -> Message:
        """
        Create a response message for a request.
        
        Args:
            request: The original request message
            result: The result payload
            success: Whether the request succeeded
            
        Returns:
            Response message
        """
        message_type = MessageType.TASK_RESULT if success else MessageType.ERROR
        
        message = Message(
            message_type=message_type,
            source_agent=self.agent_id,
            target_agent=request.source_agent,
            payload={"result": result, "success": success},
            correlation_id=request.message_id,
        )
        
        self._message_history.append(message)
        
        logger.debug(f"Created response: {message.message_id} for {request.message_id}")
        return message
    
    def create_ack(self, request: Message) -> Message:
        """Create an acknowledgment message."""
        return Message(
            message_type=MessageType.TASK_ACK,
            source_agent=self.agent_id,
            target_agent=request.source_agent,
            payload={"acknowledged": True},
            correlation_id=request.message_id,
        )
    
    def create_nack(
        self,
        request: Message,
        reason: str,
    ) -> Message:
        """Create a negative acknowledgment message."""
        return Message(
            message_type=MessageType.TASK_NACK,
            source_agent=self.agent_id,
            target_agent=request.source_agent,
            payload={"acknowledged": False, "reason": reason},
            correlation_id=request.message_id,
        )
    
    def handle_message(self, message: Message) -> Optional[Message]:
        """
        Handle an incoming message.
        
        Args:
            message: The incoming message
            
        Returns:
            Optional response message
        """
        logger.debug(f"Handling message: {message.message_id} type={message.message_type.value}")
        
        handler = self._handlers.get(message.message_type)
        if handler:
            try:
                return handler(message)
            except Exception as e:
                logger.error(f"Handler error: {e}")
                return self.create_response(message, str(e), success=False)
        else:
            logger.warning(f"No handler for message type: {message.message_type.value}")
            return None
    
    def complete_request(self, request_id: str, response: Message):
        """Mark a request as completed with its response."""
        if request_id in self._pending_requests:
            del self._pending_requests[request_id]
            logger.debug(f"Request completed: {request_id}")
    
    def get_pending_requests(self) -> List[Message]:
        """Get all pending requests."""
        return list(self._pending_requests.values())
    
    def get_message_history(
        self,
        limit: int = 100,
    ) -> List[Message]:
        """Get recent message history."""
        return self._message_history[-limit:]
