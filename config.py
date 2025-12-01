"""
MinutesX Configuration Module

Centralized configuration management using environment variables and Pydantic.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class GeminiConfig:
    """Gemini 2.5 Flash configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20"))
    temperature: float = field(default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.3")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("GEMINI_MAX_TOKENS", "4096")))
    

@dataclass
class GoogleCloudConfig:
    """Google Cloud configuration."""
    project_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    credentials_path: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    

@dataclass
class GoogleMeetConfig:
    """Google Meet integration configuration."""
    client_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_SECRET", ""))
    redirect_uri: str = field(default_factory=lambda: os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/oauth/callback"))
    scopes: List[str] = field(default_factory=lambda: [
        "https://www.googleapis.com/auth/meetings.space.readonly",
        "https://www.googleapis.com/auth/calendar.readonly",
    ])


@dataclass
class RedisConfig:
    """Redis configuration for A2A message bus."""
    url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))


@dataclass
class MemoryConfig:
    """Memory bank configuration."""
    backend: str = field(default_factory=lambda: os.getenv("MEMORY_BACKEND", "chromadb"))
    chromadb_path: str = field(default_factory=lambda: os.getenv("CHROMADB_PATH", "./data/chromadb"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/embedding-001"))
    context_window_size: int = field(default_factory=lambda: int(os.getenv("CONTEXT_WINDOW_SIZE", "32000")))


@dataclass
class IntegrationConfig:
    """External integration configuration."""
    # Slack
    slack_bot_token: str = field(default_factory=lambda: os.getenv("SLACK_BOT_TOKEN", ""))
    slack_channel: str = field(default_factory=lambda: os.getenv("SLACK_CHANNEL", "#meeting-notes"))
    
    # Jira
    jira_url: str = field(default_factory=lambda: os.getenv("JIRA_URL", ""))
    jira_email: str = field(default_factory=lambda: os.getenv("JIRA_EMAIL", ""))
    jira_api_token: str = field(default_factory=lambda: os.getenv("JIRA_API_TOKEN", ""))
    jira_project_key: str = field(default_factory=lambda: os.getenv("JIRA_PROJECT_KEY", "MEET"))


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true")
    tracing_enabled: bool = field(default_factory=lambda: os.getenv("TRACING_ENABLED", "true").lower() == "true")
    metrics_port: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "9090")))
    otel_service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "minutesx"))
    otel_endpoint: str = field(default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))


@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    parallel_timeout: float = field(default_factory=lambda: float(os.getenv("PARALLEL_AGENT_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))


@dataclass
class AppConfig:
    """Application configuration."""
    env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))


@dataclass
class Config:
    """Main configuration container."""
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    google_cloud: GoogleCloudConfig = field(default_factory=GoogleCloudConfig)
    google_meet: GoogleMeetConfig = field(default_factory=GoogleMeetConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    app: AppConfig = field(default_factory=AppConfig)
    
    def validate(self) -> bool:
        """Validate required configuration."""
        if not self.gemini.api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        return True


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
