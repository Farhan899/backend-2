"""
Agent Service - OpenAI Agents SDK Integration

This service implements the Agent Decision Hierarchy from the specification:
1. Load conversation history
2. Interpret user intent (deterministic mapping)
3. Invoke Task MCP Server tool
4. (Phase 5) Contact-7 MCP for identity enrichment
5. (Phase 5) Context-7 MCP for contextual guidance
6. Generate natural language response
7. Persist messages atomically

All operations are stateless and can run on fresh instances.
"""

from typing import Optional, Any
from uuid import UUID
from app.models import Message
from app.services.intent_mapping import IntentMapper, Intent
from app.utils.logging import logger


class AgentService:
    """Service for processing user messages through the agent decision hierarchy"""

    @staticmethod
    async def process_message(
        user_id: str,
        conversation_id: UUID,
        messages: list[Message],
        user_input: str,
        include_context: bool = False,
    ) -> tuple[str, list[dict]]:
        """
        Process a user message through the agent decision hierarchy.

        Args:
            user_id: Authenticated user ID
            conversation_id: Current conversation ID
            messages: Full message history (for context)
            user_input: New user message to process
            include_context: Include Contact-7 and Context-7 enrichment (Phase 5)

        Returns:
            Tuple of (assistant_response, tool_calls)
            - assistant_response: Natural language response from agent
            - tool_calls: Array of executed tools with parameters and results

        Agent Decision Hierarchy:
        1. Load conversation history (already done)
        2. Extract intent from user message (deterministic)
        3. Validate intent has sufficient confidence
        4. If intent unclear, ask for clarification
        5. Invoke appropriate Task MCP tool
        6. (Phase 5) Enrich with Contact-7 (user context) if include_context=True
        7. (Phase 5) Enrich with Context-7 (conversation summary) if include_context=True
        8. Generate natural language response (with personalization from Contact-7)
        9. Return (response, tool_calls) for persistence
        """

        # Step 1: Extract intent deterministically
        intent, confidence = IntentMapper.extract_intent(user_input)

        logger.log_agent_decision(
            user_id=user_id,
            conversation_id=conversation_id,
            intent=intent.value,
            confidence=confidence,
            user_input=user_input,
        )

        # Step 2: Validate confidence
        if intent == Intent.UNKNOWN or confidence < 0.5:
            response = (
                "I didn't understand your request. You can ask me to:\n"
                "- Create a task (e.g., 'add buy groceries')\n"
                "- List your tasks (e.g., 'show all tasks')\n"
                "- Complete a task (e.g., 'mark task 1 done')\n"
                "- Update a task (e.g., 'change task 1 to buy milk')\n"
                "- Delete a task (e.g., 'delete task 1')"
            )
            return response, []

        # Step 3: Prepare tool invocation
        tool_calls = []

        # Step 4: Extract parameters based on intent
        tool_name = IntentMapper.get_tool_name(intent)
        params = AgentService._extract_parameters(intent, user_input, user_id)

        # Step 5: Invoke Tool (Task MCP)
        tool_result = await AgentService._invoke_tool(
            tool_name, params, user_id, conversation_id
        )

        # Step 6 & 7: Contact-7 and Context-7 enrichment (Phase 5)
        user_context = {}
        conversation_summary = {}

        if include_context and tool_result.get("error") is None:
            # TODO: Phase 5 - Invoke Contact-7 MCP Server
            # user_context = await AgentService._invoke_contact7(user_id)

            # TODO: Phase 5 - Invoke Context-7 MCP Server
            # conversation_summary = await AgentService._invoke_context7(
            #     conversation_id, user_id
            # )

            user_context = {
                "name": "User",  # Would come from Contact-7
                "preferences": {"timezone": "UTC"},
            }
            conversation_summary = {
                "topics": ["task_creation", "task_listing"],
                "intent_summary": "User managing tasks",
            }

            logger.info(
                "Context enrichment completed",
                user_id=user_id,
                conversation_id=conversation_id,
                has_user_context=bool(user_context),
                has_conversation_summary=bool(conversation_summary),
            )

        if tool_result.get("error"):
            response = IntentMapper.get_fallback_response(intent)
            logger.log_error(
                user_id=user_id,
                conversation_id=conversation_id,
                error_type="TOOL_EXECUTION_FAILED",
                error_message=tool_result.get("error"),
                tool=tool_name,
                error_code=tool_result.get("code", 500),
            )
        else:
            # Step 8: Generate natural language response
            # Personalization informed by Contact-7 context
            response = AgentService._generate_response(
                intent, tool_result, user_input, user_context
            )

            tool_calls.append(
                {
                    "tool": tool_name,
                    "parameters": params,
                    "result": tool_result,
                }
            )

            logger.log_tool_call(
                user_id=user_id,
                conversation_id=conversation_id,
                tool_name=tool_name,
                parameters=params,
                result=tool_result,
            )

        # Step 9: Return response and tool calls
        return response, tool_calls

    @staticmethod
    def _extract_parameters(intent: Intent, user_input: str, user_id: str) -> dict:
        """
        Extract tool parameters from user input.

        This is a simplified extraction - Phase 3 will use NER and entity extraction.
        For now, we use pattern matching and heuristics.
        """
        params = {"user_id": user_id}

        if intent == Intent.ADD:
            # Extract task title from message
            # Simple heuristic: everything after "add" or "create"
            import re

            match = re.search(r"^(?:add|create|new task|remember)\s+(.+)$", user_input.lower())
            if match:
                params["title"] = match.group(1).strip()
            else:
                params["title"] = "Untitled task"

        elif intent == Intent.LIST:
            params["include_completed"] = "completed" not in user_input.lower()

        elif intent == Intent.COMPLETE:
            # Try to extract task ID
            import re

            match = re.search(r"\d+", user_input)
            if match:
                params["task_id"] = match.group()
            params["completed"] = "uncomplete" not in user_input.lower()

        elif intent == Intent.DELETE:
            import re

            match = re.search(r"\d+", user_input)
            if match:
                params["task_id"] = match.group()

        elif intent == Intent.UPDATE:
            import re

            # Try to extract task ID and new content
            match = re.search(r"(?:task\s+)?(\d+)\s+(?:to\s+)?(.+)", user_input)
            if match:
                params["task_id"] = match.group(1)
                params["title"] = match.group(2).strip()

        elif intent == Intent.GET:
            import re

            match = re.search(r"\d+", user_input)
            if match:
                params["task_id"] = match.group()

        return params

    @staticmethod
    async def _invoke_tool(
        tool_name: str, params: dict, user_id: str, conversation_id: UUID
    ) -> dict:
        """
        Invoke an MCP tool and return result.

        Using TaskMCPServer directly for now.
        """
        from mcp_servers.task_mcp.main import TaskMCPServer

        server = TaskMCPServer()

        # All task tools expect user_id in params
        params["user_id"] = user_id

        if tool_name == "add_task":
            return await server.add_task(params)
        elif tool_name == "list_tasks":
            return await server.list_tasks(params)
        elif tool_name == "get_task":
            return await server.get_task(params)
        elif tool_name == "update_task":
            return await server.update_task(params)
        elif tool_name == "delete_task":
            return await server.delete_task(params)
        elif tool_name == "complete_task":
            return await server.complete_task(params)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    @staticmethod
    def _generate_response(
        intent: Intent, tool_result: dict, user_input: str, user_context: dict = None
    ) -> str:
        """
        Generate natural language response based on tool result.

        Uses template-based approach with optional user context for personalization.
        The user_context comes from Contact-7 MCP Server (Phase 5).
        """
        if user_context is None:
            user_context = {}
        if intent == Intent.ADD:
            title = tool_result.get("title", "task")
            return f"✅ Created task: {title}"

        elif intent == Intent.LIST:
            tasks = tool_result.get("tasks", [])
            if not tasks:
                return "You don't have any tasks yet."
            task_list = "\n".join(
                [f"- {t.get('title')} {'✓' if t.get('is_completed') else ''}" for t in tasks]
            )
            return f"Here are your tasks:\n{task_list}"

        elif intent == Intent.COMPLETE:
            return f"✅ Marked task {tool_result.get('id')} as done."

        elif intent == Intent.DELETE:
            return f"🗑️ Deleted task {tool_result.get('id')}."

        elif intent == Intent.UPDATE:
            return f"✏️ Updated task: {tool_result.get('title', 'task')}"

        elif intent == Intent.GET:
            title = tool_result.get("title", "Task")
            return f"📋 **{title}**\n{tool_result.get('description', 'No description')}"

        else:
            return "Operation completed."
