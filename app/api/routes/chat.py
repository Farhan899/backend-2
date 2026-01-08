from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends, Path, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import Conversation
from app.schemas.chat import ChatRequest, ChatResponse, ToolCall
from app.services.conversation import ConversationService
from app.services.agent import AgentService
from app.api.deps import get_current_user, verify_user_id
from app.core.database import get_session
from app.utils.logging import logger

router = APIRouter(prefix="/api/{user_id}", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    user_id: Annotated[str, Depends(verify_user_id)],
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ChatResponse:
    """
    Stateless chat endpoint for AI-powered task management.

    Accepts natural language messages, reconstructs conversation history,
    and returns assistant response with tool calls.

    Args:
        user_id: Authenticated user ID from URL path (verified)
        request: ChatRequest with conversation_id, message, include_context
        session: Database session

    Returns:
        ChatResponse with conversation_id, assistant_message, tool_calls

    Raises:
        HTTPException: 400 for invalid input, 404 for not found conversation
    """
    try:
        # Load or create conversation
        if request.conversation_id:
            conversation, messages = await ConversationService.load_conversation(
                session, request.conversation_id, user_id
            )
        else:
            conversation = await ConversationService.create_conversation(session, user_id)
            messages = []

        # Persist user message
        user_message = await ConversationService.persist_user_message(
            session, conversation.id, user_id, request.message
        )

        # Step 1: Load full conversation history for agent context
        full_history = messages + [user_message]

        # Step 2: Process through Agent Decision Hierarchy
        agent_response, tool_calls = await AgentService.process_message(
            user_id=user_id,
            conversation_id=conversation.id,
            messages=full_history,
            user_input=request.message,
            include_context=request.include_context,
        )

        # Step 3: Persist assistant response with tool calls metadata
        tool_calls_json = None
        if tool_calls:
            tool_calls_json = {"tools": tool_calls}

        assistant_message = await ConversationService.persist_assistant_message(
            session,
            conversation.id,
            user_id,
            content=agent_response,
            tool_calls=tool_calls_json,
        )

        # Step 4: Commit all changes atomically
        await ConversationService.save_conversation(session)

        logger.info(
            "Chat request processed successfully",
            user_id=user_id,
            conversation_id=str(conversation.id),
            tool_count=len(tool_calls),
        )

        return ChatResponse(
            conversation_id=conversation.id,
            assistant_message=assistant_message.content,
            tool_calls=[
                ToolCall(tool=tc["tool"], parameters=tc["parameters"])
                for tc in tool_calls
            ],
        )

    except ValueError as e:
        # Conversation not found or user doesn't own it
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred processing your message",
        )
