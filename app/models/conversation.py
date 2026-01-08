from sqlmodel import Field, SQLModel
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


class Conversation(SQLModel, table=True):
    """Conversation entity representing a chat session"""
    __tablename__ = "conversations"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
