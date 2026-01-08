from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Literal

class TaskCreate(BaseModel):
    """Schema for creating a new task"""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    priority: Optional[Literal["high", "medium", "low"]] = None
    due_date: Optional[str] = None

    @field_validator('title')
    @classmethod
    def title_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip()

    @field_validator('description')
    @classmethod
    def normalize_description(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            return None
        return v

    @field_validator('due_date')
    @classmethod
    def parse_due_date(cls, v: Optional[str]) -> Optional[datetime]:
        if v is None:
            return None
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            raise ValueError('Invalid date format. Use ISO format (YYYY-MM-DD)')

class TaskUpdate(BaseModel):
    """Schema for updating an existing task"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    priority: Optional[Literal["high", "medium", "low", None]] = None
    due_date: Optional[str] = None

    @field_validator('title')
    @classmethod
    def title_not_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip() if v else v

    @field_validator('description')
    @classmethod
    def normalize_description(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            return None
        return v

    @field_validator('due_date')
    @classmethod
    def parse_due_date(cls, v: Optional[str]) -> Optional[Optional[datetime]]:
        if v is None:
            return None
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            raise ValueError('Invalid date format. Use ISO format (YYYY-MM-DD)')

class TaskResponse(BaseModel):
    """Schema for task API responses"""
    id: int
    user_id: str
    title: str
    description: Optional[str]
    is_completed: bool
    priority: Optional[str] = None
    due_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # Pydantic v2 (was orm_mode in v1)
