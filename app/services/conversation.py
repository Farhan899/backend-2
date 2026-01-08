from uuid import UUID, uuid4
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.models import Conversation, Message


class ConversationService:
    """Service for managing conversations and messages"""

    @staticmethod
    async def load_conversation(
        session: AsyncSession, conversation_id: UUID, user_id: str
    ) -> tuple[Conversation, list[Message]]:
        """
        Load a conversation with all its messages, validating user ownership.

        Args:
            session: AsyncSession for database operations
            conversation_id: UUID of the conversation to load
            user_id: User ID to verify ownership

        Returns:
            Tuple of (Conversation, list of Messages)

        Raises:
            ValueError: If conversation doesn't exist or user doesn't own it
        """
        # Load conversation
        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await session.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Verify user ownership
        if conversation.user_id != user_id:
            raise ValueError("Access denied: conversation does not belong to user")

        # Load all messages for this conversation, ordered by creation time
        messages_query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        messages_result = await session.execute(messages_query)
        messages = messages_result.scalars().all()

        return conversation, messages

    @staticmethod
    async def create_conversation(
        session: AsyncSession, user_id: str
    ) -> Conversation:
        """
        Create a new conversation for a user.

        Args:
            session: AsyncSession for database operations
            user_id: User ID for the conversation

        Returns:
            Created Conversation object
        """
        conversation = Conversation(id=uuid4(), user_id=user_id)
        session.add(conversation)
        await session.flush()
        return conversation

    @staticmethod
    async def persist_user_message(
        session: AsyncSession,
        conversation_id: UUID,
        user_id: str,
        content: str,
    ) -> Message:
        """
        Persist a user message to a conversation.

        Args:
            session: AsyncSession for database operations
            conversation_id: UUID of the conversation
            user_id: User ID (for validation)
            content: Message content

        Returns:
            Created Message object
        """
        message = Message(
            id=uuid4(),
            conversation_id=conversation_id,
            user_id=user_id,
            sender="user",
            content=content,
        )
        session.add(message)
        await session.flush()
        return message

    @staticmethod
    async def persist_assistant_message(
        session: AsyncSession,
        conversation_id: UUID,
        user_id: str,
        content: str,
        tool_calls: dict | None = None,
    ) -> Message:
        """
        Persist an assistant message to a conversation.

        Args:
            session: AsyncSession for database operations
            conversation_id: UUID of the conversation
            user_id: User ID (for validation)
            content: Message content
            tool_calls: Optional JSON object with tool call details

        Returns:
            Created Message object
        """
        message = Message(
            id=uuid4(),
            conversation_id=conversation_id,
            user_id=user_id,
            sender="assistant",
            content=content,
            tool_calls=tool_calls,
        )
        session.add(message)
        await session.flush()
        return message

    @staticmethod
    async def save_conversation(session: AsyncSession) -> None:
        """
        Commit all pending changes to the database.

        Args:
            session: AsyncSession for database operations
        """
        await session.commit()
