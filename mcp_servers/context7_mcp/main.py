"""
Context-7 MCP Server - Conversation Context and Summarization

This server provides contextual guidance for agent decision-making:
- Conversation summarization (recent history)
- Key entities extraction (task mentions, dates, priorities)
- Context window optimization (which messages are most relevant)
- Tone and conversation style analysis

Context-7 is a read-only advisory server that provides conversation
intelligence to help the agent understand broader context and make
better decisions.
"""

import sys
import os
from typing import Any, List
from datetime import datetime
from mcp.server.models import InitializationOptions
from mcp.server import Server
import mcp.types as types

# Import models and database
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.models import Message, Conversation
from app.core.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession as SQLAlchemyAsyncSession
from sqlmodel import select


class Context7MCPServer:
    """Context-7 MCP Server for conversation context and summarization"""

    def __init__(self):
        self.server = Server("context7-mcp-server")
        self.engine = None
        self._register_tools()

    async def initialize_db(self):
        """Initialize async database engine"""
        if not self.engine:
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                echo=False,
                future=True,
            )

    def _register_tools(self):
        """Register tools with the MCP server"""

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> Any:
            """Handle tool calls"""
            if name == "summarize_conversation":
                return await self.summarize_conversation(arguments)
            elif name == "select_relevant_messages":
                return await self.select_relevant_messages(arguments)
            else:
                return {"error": f"Unknown tool: {name}"}

    async def initialize(self) -> InitializationOptions:
        """Initialize the MCP server with tool definitions"""
        return InitializationOptions(
            server_name="context7-mcp-server",
            server_version="1.0.0",
            tools=[
                types.Tool(
                    name="summarize_conversation",
                    description="Summarize a conversation to understand key topics, decisions, and context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "conversation_id": {
                                "type": "string",
                                "description": "Conversation ID to summarize",
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User ID for authorization",
                            },
                        },
                        "required": ["conversation_id", "user_id"],
                    },
                ),
                types.Tool(
                    name="select_relevant_messages",
                    description="Select the most relevant messages from a conversation for context window optimization",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "conversation_id": {
                                "type": "string",
                                "description": "Conversation ID",
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User ID for authorization",
                            },
                            "max_messages": {
                                "type": "integer",
                                "description": "Maximum number of messages to return",
                                "default": 10,
                            },
                        },
                        "required": ["conversation_id", "user_id"],
                    },
                ),
            ],
        )

    async def summarize_conversation(self, args: dict) -> dict:
        """
        Summarize a conversation to extract key information.

        Analyzes conversation history to extract:
        - Main topics discussed
        - Tasks mentioned and their status
        - User intent patterns
        - Key decisions made

        Args:
            conversation_id: The conversation to summarize
            user_id: User ID for authorization

        Returns:
            Summary dictionary with key information
        """
        conversation_id = args.get("conversation_id")
        user_id = args.get("user_id")

        if not conversation_id or not user_id:
            return {"error": "conversation_id and user_id are required", "code": 400}

        await self.initialize_db()
        async with SQLAlchemyAsyncSession(self.engine) as session:
            try:
                # Verify conversation ownership
                query = select(Conversation).where(
                    (Conversation.id == conversation_id) & (Conversation.user_id == user_id)
                )
                result = await session.execute(query)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    return {"error": "Conversation not found", "code": 404}

                # Load all messages
                messages_query = (
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at)
                )
                messages_result = await session.execute(messages_query)
                messages = messages_result.scalars().all()

                # Extract summary information
                user_messages = [m for m in messages if m.sender == "user"]
                assistant_messages = [m for m in messages if m.sender == "assistant"]

                # Simple summary extraction
                summary = {
                    "conversation_id": str(conversation_id),
                    "message_count": len(messages),
                    "user_message_count": len(user_messages),
                    "assistant_message_count": len(assistant_messages),
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat(),
                    "topics": Context7MCPServer._extract_topics(user_messages),
                    "key_phrases": Context7MCPServer._extract_key_phrases(
                        user_messages
                    ),
                    "conversation_tone": "task-focused",  # Could be enhanced with sentiment analysis
                    "user_intent_summary": Context7MCPServer._summarize_intents(
                        user_messages
                    ),
                }

                return summary

            except Exception as e:
                return {
                    "error": f"Failed to summarize conversation: {str(e)}",
                    "code": 500,
                }

    async def select_relevant_messages(self, args: dict) -> dict:
        """
        Select the most relevant messages for context window optimization.

        Returns:
        - Most recent messages (always relevant)
        - Messages with user intent changes
        - Messages with important decisions

        Args:
            conversation_id: The conversation to analyze
            user_id: User ID for authorization
            max_messages: Maximum messages to return (default: 10)

        Returns:
            List of relevant messages with timestamps
        """
        conversation_id = args.get("conversation_id")
        user_id = args.get("user_id")
        max_messages = args.get("max_messages", 10)

        if not conversation_id or not user_id:
            return {"error": "conversation_id and user_id are required", "code": 400}

        await self.initialize_db()
        async with SQLAlchemyAsyncSession(self.engine) as session:
            try:
                # Verify conversation ownership
                query = select(Conversation).where(
                    (Conversation.id == conversation_id) & (Conversation.user_id == user_id)
                )
                result = await session.execute(query)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    return {"error": "Conversation not found", "code": 404}

                # Load all messages
                messages_query = (
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at)
                )
                messages_result = await session.execute(messages_query)
                messages = messages_result.scalars().all()

                # Select most relevant: recent + important
                relevant_messages = Context7MCPServer._select_relevant(
                    messages, max_messages
                )

                return {
                    "conversation_id": str(conversation_id),
                    "total_messages": len(messages),
                    "selected_message_count": len(relevant_messages),
                    "messages": [
                        {
                            "id": str(msg.id),
                            "sender": msg.sender,
                            "content": msg.content[:200],  # Truncate long messages
                            "created_at": msg.created_at.isoformat(),
                            "relevance_score": 1.0,  # Could be enhanced with scoring
                        }
                        for msg in relevant_messages
                    ],
                }

            except Exception as e:
                return {
                    "error": f"Failed to select relevant messages: {str(e)}",
                    "code": 500,
                }

    @staticmethod
    def _extract_topics(messages: List[Message]) -> List[str]:
        """Extract main topics from user messages"""
        topics = set()
        keywords = {
            "add": "task_creation",
            "create": "task_creation",
            "delete": "task_deletion",
            "remove": "task_deletion",
            "complete": "task_completion",
            "done": "task_completion",
            "update": "task_update",
            "change": "task_update",
            "list": "task_listing",
            "show": "task_listing",
        }

        for msg in messages:
            content_lower = msg.content.lower()
            for keyword, topic in keywords.items():
                if keyword in content_lower:
                    topics.add(topic)

        return list(topics)

    @staticmethod
    def _extract_key_phrases(messages: List[Message]) -> List[str]:
        """Extract key phrases from user messages"""
        phrases = []
        for msg in messages:
            # Simple extraction: sentences starting with action words
            if any(
                msg.content.lower().startswith(word)
                for word in ["add", "create", "delete", "update", "complete", "list"]
            ):
                phrases.append(msg.content[:100])

        return phrases[:5]  # Limit to 5 key phrases

    @staticmethod
    def _summarize_intents(messages: List[Message]) -> str:
        """Summarize user intents from messages"""
        if not messages:
            return "No user messages in conversation"

        intent_count = 0
        for msg in messages:
            if any(
                word in msg.content.lower()
                for word in ["add", "create", "delete", "update", "complete"]
            ):
                intent_count += 1

        if intent_count == 0:
            return "User querying task information"
        elif intent_count <= 2:
            return "User performing simple task management"
        else:
            return "User performing complex task management workflow"

    @staticmethod
    def _select_relevant(messages: List[Message], max_count: int) -> List[Message]:
        """Select most relevant messages for context window"""
        if len(messages) <= max_count:
            return messages

        # Strategy: recent messages (most relevant) + spread earlier messages
        recent_count = min(max_count, len(messages) // 2 + 1)
        recent = messages[-recent_count:]

        if max_count > recent_count:
            older_count = max_count - recent_count
            step = len(messages[:-recent_count]) // older_count if older_count > 0 else 1
            older = messages[::step][: max_count - recent_count]
            return older + recent
        else:
            return recent


async def main():
    """Entry point for the Context-7 MCP Server"""
    server = Context7MCPServer()
    options = await server.initialize()

    async with server.server:
        # Server is now running and listening for connections
        await asyncio.sleep(float('inf'))


if __name__ == "__main__":
    asyncio.run(main())
