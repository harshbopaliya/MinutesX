"""
MinutesX Tracing Module

Provides distributed tracing functionality.
"""
import functools
import time
from typing import Any, Callable, Optional
from contextlib import contextmanager

from observability.logger import get_logger


logger = get_logger(__name__)


class TraceContext:
    """Simple trace context for tracking operation spans."""
    
    def __init__(self, name: str, parent: Optional["TraceContext"] = None):
        self.name = name
        self.parent = parent
        self.start_time: float = 0
        self.end_time: float = 0
        self.attributes: dict = {}
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting span: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type:
            logger.debug(f"Span {self.name} failed after {duration:.3f}s: {exc_val}")
        else:
            logger.debug(f"Span {self.name} completed in {duration:.3f}s")
        
        return False
    
    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value
    
    @property
    def duration(self) -> float:
        """Get span duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


def trace_span(name: str) -> Callable:
    """
    Decorator to create a trace span around a function.
    
    Args:
        name: Span name
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with TraceContext(name) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with TraceContext(name) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    raise
        
        # Check if function is async
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            return async_wrapper
        return wrapper
    
    return decorator


@contextmanager
def create_span(name: str):
    """Context manager to create a trace span."""
    span = TraceContext(name)
    with span:
        yield span
