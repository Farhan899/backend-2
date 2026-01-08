from sqlmodel import Field, SQLModel
from datetime import datetime
from typing import Optional

class Task(SQLModel, table=True):
    """Task entity representing a todo item"""
    __tablename__ = "tasks"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    title: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    is_completed: bool = Field(default=False)
    priority: Optional[str] = Field(default=None, max_length=20)
    due_date: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
