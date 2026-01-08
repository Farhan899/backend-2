"""
Intent-to-Tool Mapping for AI-Native Taskify

Constitutional Section VI defines deterministic mapping from natural language
intent to specific tools. This module implements the official mapping table.

All intents are:
1. User-friendly (lowercase, natural language)
2. Unambiguous (exactly one intent per phrase)
3. Complete (covers all task operations)
4. Deterministic (identical input → identical output)
"""

from enum import Enum
from typing import Optional


class Intent(Enum):
    """Enumeration of all supported intents"""
    ADD = "add_task"
    LIST = "list_tasks"
    GET = "get_task"
    UPDATE = "update_task"
    DELETE = "delete_task"
    COMPLETE = "complete_task"
    UNKNOWN = "unknown"


class IntentMapper:
    """Maps natural language to deterministic intents"""

    # Official intent mapping table (Constitutional Section VI)
    # These mappings are exhaustive and unambiguous
    INTENT_PATTERNS = {
        # ADD TASK intents
        Intent.ADD: {
            "keywords": [
                "add",
                "create",
                "new task",
                "add task",
                "create task",
                "remember",
                "make a task",
                "i need to",
            ],
            "patterns": [
                r"^add\s+",
                r"^create\s+",
                r"^new\s+task\s+",
                r"^remember\s+",
                r"^add task",
                r"^create task",
                r"^i need to ",
            ],
        },
        # LIST TASKS intents
        Intent.LIST: {
            "keywords": [
                "list",
                "show",
                "all tasks",
                "my tasks",
                "what tasks",
                "tasks do i have",
                "what do i need",
            ],
            "patterns": [
                r"^list\s+",
                r"^show\s+",
                r"^all\s+tasks",
                r"^my\s+tasks",
                r"^what tasks",
                r"^what do i",
            ],
        },
        # GET TASK intents
        Intent.GET: {
            "keywords": [
                "get",
                "show task",
                "details",
                "tell me about",
            ],
            "patterns": [
                r"^get\s+",
                r"^show task",
                r"^details\s+",
                r"^tell me about",
            ],
        },
        # UPDATE TASK intents
        Intent.UPDATE: {
            "keywords": [
                "update",
                "edit",
                "change",
                "modify",
                "rename",
            ],
            "patterns": [
                r"^update\s+",
                r"^edit\s+",
                r"^change\s+",
                r"^modify\s+",
                r"^rename\s+",
            ],
        },
        # DELETE TASK intents
        Intent.DELETE: {
            "keywords": [
                "delete",
                "remove",
                "trash",
                "discard",
                "get rid of",
            ],
            "patterns": [
                r"^delete\s+",
                r"^remove\s+",
                r"^trash\s+",
                r"^discard\s+",
                r"^get rid of",
            ],
        },
        # COMPLETE TASK intents
        Intent.COMPLETE: {
            "keywords": [
                "complete",
                "done",
                "finish",
                "check off",
                "mark done",
                "mark as done",
            ],
            "patterns": [
                r"^complete\s+",
                r"^done\s+with",
                r"^finish\s+",
                r"^check off",
                r"^mark\s+done",
                r"^mark as done",
            ],
        },
    }

    @staticmethod
    def extract_intent(user_message: str) -> tuple[Intent, float]:
        """
        Extract intent from user message with confidence score.

        Args:
            user_message: Raw user input

        Returns:
            Tuple of (Intent, confidence_score)
            Confidence is 1.0 if pattern matches, 0.5 if keyword found

        Algorithm:
        1. Normalize message (lowercase, trim)
        2. Try pattern matching (exact, highest confidence)
        3. Fall back to keyword matching (loose, lower confidence)
        4. Return UNKNOWN if no match
        """
        import re

        message = user_message.strip().lower()

        # Try pattern matching first (highest confidence)
        for intent, config in IntentMapper.INTENT_PATTERNS.items():
            for pattern in config.get("patterns", []):
                if re.search(pattern, message, re.IGNORECASE):
                    return intent, 1.0

        # Fall back to keyword matching (lower confidence)
        for intent, config in IntentMapper.INTENT_PATTERNS.items():
            for keyword in config.get("keywords", []):
                if keyword.lower() in message.lower():
                    return intent, 0.7

        # No match found
        return Intent.UNKNOWN, 0.0

    @staticmethod
    def get_tool_name(intent: Intent) -> str:
        """Get the MCP tool name for an intent"""
        return intent.value

    @staticmethod
    def should_confirm(intent: Intent) -> bool:
        """Determine if agent should ask for confirmation before executing tool"""
        # Always confirm for delete operations
        return intent == Intent.DELETE

    @staticmethod
    def get_fallback_response(intent: Intent) -> str:
        """Get fallback response if tool fails"""
        responses = {
            Intent.ADD: "I couldn't create that task. Please try again.",
            Intent.LIST: "I couldn't retrieve your tasks. Please try again.",
            Intent.GET: "I couldn't find that task. Please try again.",
            Intent.UPDATE: "I couldn't update that task. Please try again.",
            Intent.DELETE: "I couldn't delete that task. Please try again.",
            Intent.COMPLETE: "I couldn't mark that task. Please try again.",
            Intent.UNKNOWN: "I didn't understand your request. You can ask me to create, list, complete, update, or delete tasks.",
        }
        return responses.get(intent, "Something went wrong. Please try again.")
