import json
import logging
import sys
from datetime import datetime
from typing import Any, Optional
from uuid import UUID


class StructuredLogger:
    """Structured JSON logger for agent decisions and tool calls"""

    def __init__(self, name: str = "chatbot"):
        self.logger = logging.getLogger(name)
        # Remove default handlers
        self.logger.handlers = []
        # Add JSON stream handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _serialize_value(self, value: Any) -> Any:
        """Convert non-serializable types to JSON-compatible types"""
        if isinstance(value, UUID):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return value

    def _create_log_entry(
        self,
        level: str,
        message: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        **context: Any
    ) -> str:
        """Create structured log entry as JSON"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
        }

        if user_id:
            entry["user_id"] = user_id
        if conversation_id:
            entry["conversation_id"] = str(conversation_id)

        # Add context fields
        for key, value in context.items():
            entry[key] = self._serialize_value(value)

        return json.dumps(entry)

    def info(
        self,
        message: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        **context: Any
    ) -> None:
        """Log info level with structured context"""
        log_entry = self._create_log_entry(
            "INFO", message, user_id, conversation_id, **context
        )
        self.logger.info(log_entry)

    def error(
        self,
        message: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        **context: Any
    ) -> None:
        """Log error level with structured context"""
        log_entry = self._create_log_entry(
            "ERROR", message, user_id, conversation_id, **context
        )
        self.logger.error(log_entry)

    def warning(
        self,
        message: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        **context: Any
    ) -> None:
        """Log warning level with structured context"""
        log_entry = self._create_log_entry(
            "WARNING", message, user_id, conversation_id, **context
        )
        self.logger.warning(log_entry)

    def debug(
        self,
        message: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        **context: Any
    ) -> None:
        """Log debug level with structured context"""
        log_entry = self._create_log_entry(
            "DEBUG", message, user_id, conversation_id, **context
        )
        self.logger.debug(log_entry)

    def log_agent_decision(
        self,
        user_id: str,
        conversation_id: UUID,
        intent: str,
        confidence: float,
        **context: Any
    ) -> None:
        """Log agent intent decision"""
        self.info(
            "Agent intent extracted",
            user_id=user_id,
            conversation_id=conversation_id,
            intent=intent,
            confidence=confidence,
            **context
        )

    def log_tool_call(
        self,
        user_id: str,
        conversation_id: UUID,
        tool_name: str,
        parameters: dict,
        result: Any = None,
        latency_ms: Optional[float] = None,
        **context: Any
    ) -> None:
        """Log tool execution"""
        self.info(
            f"Tool executed: {tool_name}",
            user_id=user_id,
            conversation_id=conversation_id,
            tool=tool_name,
            parameters=parameters,
            result=result,
            latency_ms=latency_ms,
            **context
        )

    def log_error(
        self,
        user_id: str,
        conversation_id: Optional[UUID],
        error_type: str,
        error_message: str,
        **context: Any
    ) -> None:
        """Log error with context (without exposing secrets)"""
        self.error(
            f"Error: {error_type}",
            user_id=user_id,
            conversation_id=conversation_id,
            error_type=error_type,
            error_message=error_message,
            **context
        )


# Global logger instance
logger = StructuredLogger("chatbot")
