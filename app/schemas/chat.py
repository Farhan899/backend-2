from pydantic import BaseModel, Field
from typing import Optional, Any, List
from uuid import UUID


class ChatRequest(BaseModel):
    """Schema for chat endpoint request"""
    conversation_id: Optional[UUID] = Field(None, description="Existing conversation ID, or null to create new")
    message: str = Field(..., min_length=1, description="User message")
    include_context: bool = Field(False, description="Include task context in agent decision-making")


class ToolCall(BaseModel):
    """Schema for a tool call in the response"""
    tool: str
    parameters: dict[str, Any]


class ChatResponse(BaseModel):
    """Schema for chat endpoint response"""
    conversation_id: UUID
    assistant_message: str
    tool_calls: List[ToolCall] = Field(default_factory=list)

    class Config:
        from_attributes = True
