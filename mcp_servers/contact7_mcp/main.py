"""
Contact-7 MCP Server - User Identity and Profile Enrichment

This server provides context about the authenticated user to enrich agent decisions:
- User profile (name, email, preferences)
- Team/organization information
- User settings and preferences

Contact-7 is a read-only advisory server that provides identity context
for personalizing agent responses and decisions.
"""

import sys
import os
from typing import Any
from mcp.server.models import InitializationOptions
from mcp.server import Server
import mcp.types as types

# Import models and database
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.models import User
from app.core.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession as SQLAlchemyAsyncSession
from sqlmodel import select


class Contact7MCPServer:
    """Contact-7 MCP Server for user identity enrichment"""

    def __init__(self):
        self.server = Server("contact7-mcp-server")
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
            if name == "get_user_context":
                return await self.get_user_context(arguments)
            else:
                return {"error": f"Unknown tool: {name}"}

    async def initialize(self) -> InitializationOptions:
        """Initialize the MCP server with tool definitions"""
        return InitializationOptions(
            server_name="contact7-mcp-server",
            server_version="1.0.0",
            tools=[
                types.Tool(
                    name="get_user_context",
                    description="Get user profile and identity context for personalizing agent responses",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User ID to retrieve context for",
                            },
                        },
                        "required": ["user_id"],
                    },
                ),
            ],
        )

    async def get_user_context(self, args: dict) -> dict:
        """
        Retrieve user context for agent personalization.

        Returns user profile information that helps the agent:
        - Personalize responses with user's name
        - Understand user's preferences
        - Provide culturally appropriate responses
        - Track user's communication patterns

        Args:
            user_id: The user ID to retrieve context for

        Returns:
            User context dictionary with profile info
        """
        user_id = args.get("user_id")

        if not user_id:
            return {"error": "user_id is required", "code": 400}

        await self.initialize_db()
        async with SQLAlchemyAsyncSession(self.engine) as session:
            try:
                # Load user from database
                query = select(User).where(User.id == user_id)
                result = await session.execute(query)
                user = result.scalar_one_or_none()

                if not user:
                    return {
                        "error": "User not found",
                        "code": 404,
                    }

                # Return user context
                return {
                    "user_id": user.id,
                    "name": user.name or "User",
                    "email": user.email,
                    "email_verified": user.emailVerified,
                    "created_at": user.createdAt.isoformat(),
                    "preferences": {
                        "timezone": "UTC",  # Could be extended to store user preferences
                        "language": "en",
                        "task_notification": True,
                    },
                    "context": {
                        "user_type": "individual",
                        "account_status": "active",
                        "task_count_estimate": None,  # Will be populated by agent
                    },
                }

            except Exception as e:
                return {
                    "error": f"Failed to retrieve user context: {str(e)}",
                    "code": 500,
                }


async def main():
    """Entry point for the Contact-7 MCP Server"""
    server = Contact7MCPServer()
    options = await server.initialize()

    async with server.server:
        # Server is now running and listening for connections
        await asyncio.sleep(float('inf'))


if __name__ == "__main__":
    asyncio.run(main())
