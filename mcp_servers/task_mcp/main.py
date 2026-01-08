"""
Task MCP Server - Authoritative task state management via MCP protocol

This server implements 6 tools for full CRUD operations on tasks:
- add_task: Create a new task
- list_tasks: List all tasks for a user
- get_task: Retrieve a single task
- update_task: Update task title/description
- delete_task: Permanently delete a task
- complete_task: Mark task as completed/uncompleted

All operations enforce user ownership validation (user_id from context).
"""

import json
import sys
from typing import Any
from datetime import datetime
from sqlmodel import select, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession as SQLAlchemyAsyncSession
from mcp.server.models import InitializationOptions
from mcp.server import Server
import mcp.types as types

# Import models and database
import asyncio
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.models import Task
from app.core.config import settings


class TaskMCPServer:
    """Task MCP Server implementation"""

    def __init__(self):
        self.server = Server("task-mcp-server")
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
        """Register all task tools with the MCP server"""

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> Any:
            """Handle tool calls"""
            if name == "add_task":
                return await self.add_task(arguments)
            elif name == "list_tasks":
                return await self.list_tasks(arguments)
            elif name == "get_task":
                return await self.get_task(arguments)
            elif name == "update_task":
                return await self.update_task(arguments)
            elif name == "delete_task":
                return await self.delete_task(arguments)
            elif name == "complete_task":
                return await self.complete_task(arguments)
            else:
                return {"error": f"Unknown tool: {name}"}

    async def initialize(self) -> InitializationOptions:
        """Initialize the MCP server with tool definitions"""
        return InitializationOptions(
            server_name="task-mcp-server",
            server_version="1.0.0",
            tools=[
                types.Tool(
                    name="add_task",
                    description="Create a new task for the authenticated user",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "User ID (from JWT)"},
                            "title": {"type": "string", "description": "Task title (1-200 chars)"},
                            "description": {"type": "string", "description": "Optional description (max 2000 chars)"},
                        },
                        "required": ["user_id", "title"],
                    },
                ),
                types.Tool(
                    name="list_tasks",
                    description="List all tasks for the authenticated user",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "User ID"},
                            "include_completed": {"type": "boolean", "description": "Include completed tasks", "default": True},
                        },
                        "required": ["user_id"],
                    },
                ),
                types.Tool(
                    name="get_task",
                    description="Retrieve a single task by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "User ID"},
                            "task_id": {"type": "string", "description": "Task ID to retrieve"},
                        },
                        "required": ["user_id", "task_id"],
                    },
                ),
                types.Tool(
                    name="update_task",
                    description="Update task title and/or description",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "User ID"},
                            "task_id": {"type": "string", "description": "Task ID to update"},
                            "title": {"type": "string", "description": "New title (optional)"},
                            "description": {"type": "string", "description": "New description (optional)"},
                        },
                        "required": ["user_id", "task_id"],
                    },
                ),
                types.Tool(
                    name="delete_task",
                    description="Delete a task permanently",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "User ID"},
                            "task_id": {"type": "string", "description": "Task ID to delete"},
                        },
                        "required": ["user_id", "task_id"],
                    },
                ),
                types.Tool(
                    name="complete_task",
                    description="Mark a task as completed or uncompleted",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "User ID"},
                            "task_id": {"type": "string", "description": "Task ID"},
                            "completed": {"type": "boolean", "description": "Completion status", "default": True},
                        },
                        "required": ["user_id", "task_id"],
                    },
                ),
            ],
        )

    async def add_task(self, args: dict) -> dict:
        """Create a new task"""
        user_id = args.get("user_id")
        title = args.get("title", "").strip()
        description = args.get("description", "").strip() or None

        # Validation
        if not title:
            return {"error": "Task title cannot be empty", "code": 400}
        if len(title) > 200:
            return {"error": "Task title cannot exceed 200 characters", "code": 400}
        if description and len(description) > 2000:
            return {"error": "Task description cannot exceed 2000 characters", "code": 400}

        await self.initialize_db()
        async with SQLAlchemyAsyncSession(self.engine) as session:
            try:
                task = Task(
                    user_id=user_id,
                    title=title,
                    description=description,
                    is_completed=False,
                )
                session.add(task)
                await session.commit()
                await session.refresh(task)

                return {
                    "id": task.id,
                    "user_id": task.user_id,
                    "title": task.title,
                    "description": task.description,
                    "is_completed": task.is_completed,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                }
            except Exception as e:
                return {"error": f"Failed to create task: {str(e)}", "code": 500}

    async def list_tasks(self, args: dict) -> dict:
        """List all tasks for a user"""
        user_id = args.get("user_id")
        include_completed = args.get("include_completed", True)

        await self.initialize_db()
        async with SQLAlchemyAsyncSession(self.engine) as session:
            try:
                query = select(Task).where(Task.user_id == user_id)

                if not include_completed:
                    query = query.where(Task.is_completed == False)

                query = query.order_by(Task.created_at.desc())
                result = await session.execute(query)
                tasks = result.scalars().all()

                return {
                    "tasks": [
                        {
                            "id": task.id,
                            "user_id": task.user_id,
                            "title": task.title,
                            "description": task.description,
                            "is_completed": task.is_completed,
                            "created_at": task.created_at.isoformat(),
                            "updated_at": task.updated_at.isoformat(),
                        }
                        for task in tasks
                    ]
                }
            except Exception as e:
                return {"error": f"Failed to list tasks: {str(e)}", "code": 500}

    async def get_task(self, args: dict) -> dict:
        """Retrieve a single task"""
        user_id = args.get("user_id")
        task_id = args.get("task_id")

        await self.initialize_db()
        async with SQLAlchemyAsyncSession(self.engine) as session:
            try:
                query = select(Task).where(
                    (Task.id == int(task_id)) & (Task.user_id == user_id)
                )
                result = await session.execute(query)
                task = result.scalar_one_or_none()

                if not task:
                    return {"error": "Task not found", "code": 404}

                return {
                    "id": task.id,
                    "user_id": task.user_id,
                    "title": task.title,
                    "description": task.description,
                    "is_completed": task.is_completed,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                }
            except ValueError:
                return {"error": "Invalid task ID format", "code": 400}
            except Exception as e:
                return {"error": f"Failed to get task: {str(e)}", "code": 500}

    async def update_task(self, args: dict) -> dict:
        """Update a task"""
        user_id = args.get("user_id")
        task_id = args.get("task_id")
        title = args.get("title")
        description = args.get("description")

        await self.initialize_db()
        async with SQLAlchemyAsyncSession(self.engine) as session:
            try:
                query = select(Task).where(
                    (Task.id == int(task_id)) & (Task.user_id == user_id)
                )
                result = await session.execute(query)
                task = result.scalar_one_or_none()

                if not task:
                    return {"error": "Task not found", "code": 404}

                # Update fields if provided
                if title is not None:
                    title = title.strip()
                    if not title:
                        return {"error": "Task title cannot be empty", "code": 400}
                    if len(title) > 200:
                        return {"error": "Task title cannot exceed 200 characters", "code": 400}
                    task.title = title

                if description is not None:
                    description = description.strip() or None
                    if description and len(description) > 2000:
                        return {"error": "Task description cannot exceed 2000 characters", "code": 400}
                    task.description = description

                task.updated_at = datetime.utcnow()
                session.add(task)
                await session.commit()
                await session.refresh(task)

                return {
                    "id": task.id,
                    "user_id": task.user_id,
                    "title": task.title,
                    "description": task.description,
                    "is_completed": task.is_completed,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                }
            except ValueError:
                return {"error": "Invalid task ID format", "code": 400}
            except Exception as e:
                return {"error": f"Failed to update task: {str(e)}", "code": 500}

    async def delete_task(self, args: dict) -> dict:
        """Delete a task"""
        user_id = args.get("user_id")
        task_id = args.get("task_id")

        await self.initialize_db()
        async with SQLAlchemyAsyncSession(self.engine) as session:
            try:
                query = select(Task).where(
                    (Task.id == int(task_id)) & (Task.user_id == user_id)
                )
                result = await session.execute(query)
                task = result.scalar_one_or_none()

                if not task:
                    return {"error": "Task not found", "code": 404}

                await session.delete(task)
                await session.commit()

                return {
                    "success": True,
                    "message": "Task deleted successfully",
                    "task_id": task_id,
                }
            except ValueError:
                return {"error": "Invalid task ID format", "code": 400}
            except Exception as e:
                return {"error": f"Failed to delete task: {str(e)}", "code": 500}

    async def complete_task(self, args: dict) -> dict:
        """Mark a task as completed or uncompleted"""
        user_id = args.get("user_id")
        task_id = args.get("task_id")
        completed = args.get("completed", True)

        await self.initialize_db()
        async with SQLAlchemyAsyncSession(self.engine) as session:
            try:
                query = select(Task).where(
                    (Task.id == int(task_id)) & (Task.user_id == user_id)
                )
                result = await session.execute(query)
                task = result.scalar_one_or_none()

                if not task:
                    return {"error": "Task not found", "code": 404}

                task.is_completed = completed
                task.updated_at = datetime.utcnow()
                session.add(task)
                await session.commit()
                await session.refresh(task)

                return {
                    "id": task.id,
                    "user_id": task.user_id,
                    "title": task.title,
                    "description": task.description,
                    "is_completed": task.is_completed,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                }
            except ValueError:
                return {"error": "Invalid task ID format", "code": 400}
            except Exception as e:
                return {"error": f"Failed to complete task: {str(e)}", "code": 500}


async def main():
    """Entry point for the Task MCP Server"""
    server = TaskMCPServer()
    options = await server.initialize()

    async with server.server:
        # Server is now running and listening for connections
        await asyncio.sleep(float('inf'))


if __name__ == "__main__":
    asyncio.run(main())
