from sqlmodel import Field, SQLModel, JSON
from datetime import datetime
from typing import Optional, Any
from uuid import UUID, uuid4


class Message(SQLModel, table=True):
    """Message entity representing a chat message"""
    __tablename__ = "messages"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id", index=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    sender: str = Field(default="user")  # "user" or "assistant"
    content: str = Field()  # Main message text
    tool_calls: Optional[Any] = Field(default=None, sa_type=JSON)  # JSON array of tool calls
    created_at: datetime = Field(default_factory=datetime.utcnow)
