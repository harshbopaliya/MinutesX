"""
MinutesX Metrics Module

Provides metrics collection for monitoring.
"""
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class MetricsCollector:
    """Simple metrics collector for tracking application metrics."""
    
    meetings_processed: int = 0
    meetings_failed: int = 0
    total_processing_time: float = 0.0
    agent_executions: Dict[str, int] = field(default_factory=dict)
    _start_time: datetime = field(default_factory=datetime.utcnow)
    
    def record_meeting_processed(
        self,
        meeting_id: str,
        duration: float,
        status: str = "success",
    ):
        """Record a meeting processing event."""
        if status == "success":
            self.meetings_processed += 1
            self.total_processing_time += duration
        else:
            self.meetings_failed += 1
    
    def record_agent_execution(self, agent_name: str, status: str = "success"):
        """Record an agent execution."""
        key = f"{agent_name}_{status}"
        self.agent_executions[key] = self.agent_executions.get(key, 0) + 1
    
    def record_transcription_chunk(self, duration: float, word_count: int):
        """Record a transcription chunk."""
        pass  # Placeholder for transcription metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics statistics."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        avg_processing_time = (
            self.total_processing_time / self.meetings_processed
            if self.meetings_processed > 0
            else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "meetings_processed": self.meetings_processed,
            "meetings_failed": self.meetings_failed,
            "success_rate": (
                self.meetings_processed / (self.meetings_processed + self.meetings_failed)
                if (self.meetings_processed + self.meetings_failed) > 0
                else 1.0
            ),
            "avg_processing_time": avg_processing_time,
            "agent_executions": self.agent_executions,
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()
