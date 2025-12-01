"""
MessageBus - A2A message routing and delivery.

Provides message routing between agents using:
- In-memory routing for local deployment
- Redis for distributed deployment
"""
import asyncio
import json
from typing import Any, Callable, Dict, List, Optional
from threading import Lock

from a2a.protocol import Message, MessageType
from config import config
from observability.logger import get_logger


logger = get_logger(__name__)


class MessageBus:
    """
    Message bus for A2A communication.
    
    Routes messages between agents with support for:
    - Synchronous delivery
    - Async delivery
    - Redis-backed distributed routing
    """
    
    def __init__(self, use_redis: bool = False):
        self.use_redis = use_redis
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queues: Dict[str, List[Message]] = {}
        self._lock = Lock()
        self._redis_client = None
        
        if use_redis:
            self._init_redis()
        
        logger.info(f"MessageBus initialized (redis={use_redis})")
    
    def _init_redis(self):
        """Initialize Redis client for distributed messaging."""
        try:
            import redis
            
            self._redis_client = redis.from_url(config.redis.url)
            self._redis_client.ping()
            logger.info("Redis connection established")
        except ImportError:
            logger.warning("Redis not installed, using in-memory bus")
            self.use_redis = False
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.use_redis = False
    
    def subscribe(
        self,
        agent_id: str,
        handler: Callable[[Message], None],
    ):
        """
        Subscribe an agent to receive messages.
        
        Args:
            agent_id: Agent identifier
            handler: Callback to handle messages
        """
        with self._lock:
            if agent_id not in self._subscribers:
                self._subscribers[agent_id] = []
            self._subscribers[agent_id].append(handler)
        
        logger.debug(f"Agent subscribed: {agent_id}")
    
    def unsubscribe(self, agent_id: str):
        """Unsubscribe an agent from messages."""
        with self._lock:
            if agent_id in self._subscribers:
                del self._subscribers[agent_id]
        
        logger.debug(f"Agent unsubscribed: {agent_id}")
    
    def publish(self, message: Message) -> bool:
        """
        Publish a message to the bus.
        
        Args:
            message: Message to publish
            
        Returns:
            Whether the message was delivered
        """
        target = message.target_agent
        
        logger.debug(f"Publishing message: {message.message_id} -> {target}")
        
        if self.use_redis and self._redis_client:
            return self._publish_redis(message)
        else:
            return self._publish_local(message)
    
    def _publish_local(self, message: Message) -> bool:
        """Publish message locally."""
        target = message.target_agent
        
        # Queue the message
        with self._lock:
            if target not in self._queues:
                self._queues[target] = []
            self._queues[target].append(message)
        
        # Deliver to subscribers
        if target in self._subscribers:
            for handler in self._subscribers[target]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
            return True
        
        return False
    
    def _publish_redis(self, message: Message) -> bool:
        """Publish message via Redis."""
        try:
            channel = f"a2a:{message.target_agent}"
            self._redis_client.publish(
                channel,
                json.dumps(message.to_dict())
            )
            return True
        except Exception as e:
            logger.error(f"Redis publish failed: {e}")
            return self._publish_local(message)
    
    def get_messages(
        self,
        agent_id: str,
        clear: bool = True,
    ) -> List[Message]:
        """
        Get queued messages for an agent.
        
        Args:
            agent_id: Agent identifier
            clear: Whether to clear the queue
            
        Returns:
            List of queued messages
        """
        with self._lock:
            messages = self._queues.get(agent_id, [])
            if clear:
                self._queues[agent_id] = []
            return messages
    
    def send_request(
        self,
        source_agent: str,
        target_agent: str,
        payload: Dict[str, Any],
    ) -> Message:
        """
        Send a task request message.
        
        Args:
            source_agent: Source agent ID
            target_agent: Target agent ID
            payload: Request payload
            
        Returns:
            The sent message
        """
        message = Message(
            message_type=MessageType.TASK_REQUEST,
            source_agent=source_agent,
            target_agent=target_agent,
            payload=payload,
        )
        
        self.publish(message)
        return message
    
    def send_response(
        self,
        source_agent: str,
        request: Message,
        result: Any,
        success: bool = True,
    ) -> Message:
        """
        Send a response to a request.
        
        Args:
            source_agent: Responding agent ID
            request: Original request message
            result: Response result
            success: Whether request succeeded
            
        Returns:
            The sent response message
        """
        message = Message(
            message_type=MessageType.TASK_RESULT if success else MessageType.ERROR,
            source_agent=source_agent,
            target_agent=request.source_agent,
            payload={"result": result, "success": success},
            correlation_id=request.message_id,
        )
        
        self.publish(message)
        return message
    
    async def listen_redis(self, agent_id: str, handler: Callable):
        """
        Listen for Redis messages asynchronously.
        
        Args:
            agent_id: Agent to listen for
            handler: Message handler callback
        """
        if not self.use_redis or not self._redis_client:
            logger.warning("Redis not available for listening")
            return
        
        pubsub = self._redis_client.pubsub()
        channel = f"a2a:{agent_id}"
        
        try:
            pubsub.subscribe(channel)
            logger.info(f"Listening on Redis channel: {channel}")
            
            while True:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    msg = Message.from_dict(data)
                    handler(msg)
                
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
        finally:
            pubsub.unsubscribe(channel)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            "use_redis": self.use_redis,
            "subscriber_count": len(self._subscribers),
            "queue_sizes": {k: len(v) for k, v in self._queues.items()},
        }
