"""
MinutesX Logging Module

Provides structured logging with JSON format support.
"""
import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in ("name", "msg", "args", "created", "filename", "funcName",
                             "levelname", "levelno", "lineno", "module", "msecs",
                             "pathname", "process", "processName", "relativeCreated",
                             "stack_info", "exc_info", "exc_text", "thread", "threadName",
                             "message", "taskName"):
                    log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        
        # Format extra fields
        extra = ""
        if hasattr(record, "__dict__"):
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in ("name", "msg", "args", "created", "filename", "funcName",
                             "levelname", "levelno", "lineno", "module", "msecs",
                             "pathname", "process", "processName", "relativeCreated",
                             "stack_info", "exc_info", "exc_text", "thread", "threadName",
                             "message", "taskName"):
                    extra_fields[key] = value
            if extra_fields:
                extra = f" | {extra_fields}"
        
        return f"{color}[{timestamp}] {record.levelname:8}{self.RESET} {record.name}: {record.getMessage()}{extra}"


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set level
        log_level = level or "INFO"
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ConsoleFormatter())
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


def log_agent_execution(agent_name: str, status: str, details: Optional[Dict[str, Any]] = None):
    """
    Log agent execution status.
    
    Args:
        agent_name: Name of the agent
        status: Execution status (success, failure)
        details: Optional additional details
    """
    logger = get_logger("agent_execution")
    
    log_data = {
        "agent": agent_name,
        "status": status,
        **(details or {}),
    }
    
    if status == "success":
        logger.info(f"Agent {agent_name} completed", extra=log_data)
    else:
        logger.error(f"Agent {agent_name} failed", extra=log_data)
