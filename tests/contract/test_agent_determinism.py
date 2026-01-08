"""
Contract Tests for Agent Determinism

Tests that identical user intents always produce identical tool invocations
(Constitutional Section III requirement).

These tests verify the contract between IntentMapper and AgentService:
- Same input → Same intent extracted
- Same intent → Same tool invoked
- Same tool → Same parameters extracted
- Same parameters → Same result
"""

import pytest
from uuid import uuid4
from app.services.intent_mapping import IntentMapper, Intent
from app.services.agent import AgentService
from app.models import Message


class TestIntentDeterminism:
    """Tests that intent extraction is deterministic"""

    def test_identical_messages_produce_identical_intents(self):
        """Test that processing identical messages always produces the same intent"""
        message = "add buy groceries"

        intents_and_confidences = []
        for _ in range(5):
            intent, confidence = IntentMapper.extract_intent(message)
            intents_and_confidences.append((intent, confidence))

        # All results should be identical
        for result in intents_and_confidences[1:]:
            assert result == intents_and_confidences[0]

    def test_different_users_same_message_identical_intents(self):
        """Test that different users get same intent for same message"""
        message = "add task"
        user_ids = ["user-1", "user-2", "user-3"]

        results = {}
        for user_id in user_ids:
            intent, confidence = IntentMapper.extract_intent(message)
            results[user_id] = (intent, confidence)

        # All users should get identical intent
        first_result = results["user-1"]
        for user_id in user_ids[1:]:
            assert results[user_id] == first_result

    def test_add_intent_deterministic_across_variations(self):
        """Test add intent detection is deterministic across message variations"""
        test_cases = [
            "add buy groceries",
            "add task buy groceries",
            "create task: buy groceries",
            "remember to buy groceries",
        ]

        results = {}
        for message in test_cases:
            # Extract intent multiple times
            intents = [IntentMapper.extract_intent(message)[0] for _ in range(3)]
            # All should be identical
            assert intents[0] == intents[1] == intents[2]
            results[message] = intents[0]

        # All add-like intents should map to ADD
        assert all(intent == Intent.ADD for intent in results.values())

    def test_list_intent_deterministic(self):
        """Test list intent extraction is deterministic"""
        test_cases = [
            "show all tasks",
            "list my tasks",
            "what tasks do i have",
            "my tasks",
        ]

        for message in test_cases:
            intents = [IntentMapper.extract_intent(message)[0] for _ in range(3)]
            assert intents[0] == intents[1] == intents[2] == Intent.LIST

    def test_complete_intent_deterministic(self):
        """Test complete intent extraction is deterministic"""
        test_cases = [
            "complete task 1",
            "mark task 1 done",
            "finish task 1",
            "check off task 1",
        ]

        for message in test_cases:
            intents = [IntentMapper.extract_intent(message)[0] for _ in range(3)]
            assert intents[0] == intents[1] == intents[2] == Intent.COMPLETE

    def test_delete_intent_deterministic(self):
        """Test delete intent extraction is deterministic"""
        test_cases = [
            "delete task 1",
            "remove task 1",
            "trash task 1",
        ]

        for message in test_cases:
            intents = [IntentMapper.extract_intent(message)[0] for _ in range(3)]
            assert intents[0] == intents[1] == intents[2] == Intent.DELETE

    def test_update_intent_deterministic(self):
        """Test update intent extraction is deterministic"""
        test_cases = [
            "update task 1",
            "edit task 1",
            "change task 1",
            "modify task 1",
        ]

        for message in test_cases:
            intents = [IntentMapper.extract_intent(message)[0] for _ in range(3)]
            assert intents[0] == intents[1] == intents[2] == Intent.UPDATE

    def test_case_insensitive_still_deterministic(self):
        """Test that case-insensitive matching is deterministic"""
        message_pairs = [
            ("add task", "ADD TASK"),
            ("list tasks", "LIST TASKS"),
            ("Add Task", "add task"),
        ]

        for lower, upper in message_pairs:
            lower_intents = [
                IntentMapper.extract_intent(lower)[0] for _ in range(3)
            ]
            upper_intents = [
                IntentMapper.extract_intent(upper)[0] for _ in range(3)
            ]

            # Both should be deterministic within themselves
            assert lower_intents[0] == lower_intents[1] == lower_intents[2]
            assert upper_intents[0] == upper_intents[1] == upper_intents[2]

            # Both should produce the same intent
            assert lower_intents[0] == upper_intents[0]


class TestToolMappingDeterminism:
    """Tests that intents map to tools deterministically"""

    def test_intent_to_tool_mapping_fixed(self):
        """Test that intent→tool mapping is static and fixed"""
        intent_tool_mapping = {
            Intent.ADD: "add_task",
            Intent.LIST: "list_tasks",
            Intent.GET: "get_task",
            Intent.UPDATE: "update_task",
            Intent.DELETE: "delete_task",
            Intent.COMPLETE: "complete_task",
        }

        # Verify mapping across multiple calls
        for _ in range(5):
            for intent, expected_tool in intent_tool_mapping.items():
                tool_name = IntentMapper.get_tool_name(intent)
                assert tool_name == expected_tool

    def test_tool_names_never_change(self):
        """Test that tool names are immutable"""
        tools_from_call_1 = {
            IntentMapper.get_tool_name(intent): intent
            for intent in Intent
            if intent != Intent.UNKNOWN
        }

        tools_from_call_2 = {
            IntentMapper.get_tool_name(intent): intent
            for intent in Intent
            if intent != Intent.UNKNOWN
        }

        assert tools_from_call_1 == tools_from_call_2


class TestParameterExtractionDeterminism:
    """Tests that parameter extraction from identical messages is deterministic"""

    @pytest.mark.asyncio
    async def test_add_parameter_extraction_deterministic(self):
        """Test that add task parameters are extracted identically"""
        message = "add buy milk at the store"
        user_id = "test-user-123"

        params_list = []
        for _ in range(3):
            params = AgentService._extract_parameters(
                Intent.ADD, message, user_id
            )
            params_list.append(params)

        # All should be identical
        assert params_list[0] == params_list[1] == params_list[2]

    @pytest.mark.asyncio
    async def test_complete_parameter_extraction_deterministic(self):
        """Test that complete task parameters are extracted identically"""
        message = "complete task 5"
        user_id = "test-user-123"

        params_list = []
        for _ in range(3):
            params = AgentService._extract_parameters(
                Intent.COMPLETE, message, user_id
            )
            params_list.append(params)

        assert params_list[0] == params_list[1] == params_list[2]
        assert params_list[0].get("task_id") == "5"

    @pytest.mark.asyncio
    async def test_delete_parameter_extraction_deterministic(self):
        """Test that delete task parameters are extracted identically"""
        message = "delete task 3"
        user_id = "test-user-123"

        params_list = []
        for _ in range(3):
            params = AgentService._extract_parameters(
                Intent.DELETE, message, user_id
            )
            params_list.append(params)

        assert params_list[0] == params_list[1] == params_list[2]
        assert params_list[0].get("task_id") == "3"

    @pytest.mark.asyncio
    async def test_update_parameter_extraction_deterministic(self):
        """Test that update task parameters are extracted identically"""
        message = "update task 2 to new title"
        user_id = "test-user-123"

        params_list = []
        for _ in range(3):
            params = AgentService._extract_parameters(
                Intent.UPDATE, message, user_id
            )
            params_list.append(params)

        assert params_list[0] == params_list[1] == params_list[2]
        assert params_list[0].get("task_id") == "2"
        assert "new title" in params_list[0].get("title", "").lower()


class TestEndToEndDeterminism:
    """Tests full end-to-end flow determinism"""

    @pytest.mark.asyncio
    async def test_identical_requests_identical_responses(self):
        """Test that identical requests produce identical tool invocations"""
        user_id = "test-user-123"
        conversation_id = uuid4()
        messages: list[Message] = []
        user_input = "add buy groceries"

        # Process the same message twice
        response_1, tools_1 = await AgentService.process_message(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=messages,
            user_input=user_input,
            include_context=False,
        )

        response_2, tools_2 = await AgentService.process_message(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=messages,
            user_input=user_input,
            include_context=False,
        )

        # Same message should invoke same tool
        assert len(tools_1) == len(tools_2)
        if tools_1:
            assert tools_1[0]["tool"] == tools_2[0]["tool"]
            assert tools_1[0]["parameters"] == tools_2[0]["parameters"]

    @pytest.mark.asyncio
    async def test_determinism_across_conversations(self):
        """Test that intent mapping is deterministic across different conversations"""
        user_id = "test-user-123"
        message = "add important task"

        # Process in conversation 1
        conversation_1 = uuid4()
        response_1, tools_1 = await AgentService.process_message(
            user_id=user_id,
            conversation_id=conversation_1,
            messages=[],
            user_input=message,
            include_context=False,
        )

        # Process in conversation 2
        conversation_2 = uuid4()
        response_2, tools_2 = await AgentService.process_message(
            user_id=user_id,
            conversation_id=conversation_2,
            messages=[],
            user_input=message,
            include_context=False,
        )

        # Both should invoke the same tool
        assert len(tools_1) == len(tools_2)
        if tools_1 and tools_2:
            assert tools_1[0]["tool"] == tools_2[0]["tool"]

    @pytest.mark.asyncio
    async def test_determinism_across_users(self):
        """Test that intent mapping is deterministic across different users"""
        message = "list my tasks"
        conversation_id = uuid4()

        # User 1
        response_1, tools_1 = await AgentService.process_message(
            user_id="user-1",
            conversation_id=conversation_id,
            messages=[],
            user_input=message,
            include_context=False,
        )

        # User 2
        response_2, tools_2 = await AgentService.process_message(
            user_id="user-2",
            conversation_id=conversation_id,
            messages=[],
            user_input=message,
            include_context=False,
        )

        # Both users should invoke the same tool for the same intent
        assert len(tools_1) == len(tools_2)
        if tools_1 and tools_2:
            assert tools_1[0]["tool"] == tools_2[0]["tool"]


class TestFallbackResponseDeterminism:
    """Tests that fallback responses are deterministic"""

    def test_fallback_responses_consistent(self):
        """Test that fallback responses for same intent are consistent"""
        for intent in Intent:
            if intent == Intent.UNKNOWN:
                continue

            responses = []
            for _ in range(3):
                response = IntentMapper.get_fallback_response(intent)
                responses.append(response)

            # All responses should be identical
            assert responses[0] == responses[1] == responses[2]

    def test_fallback_responses_not_random(self):
        """Test that fallback responses are deterministic, not random"""
        intent = Intent.ADD

        response_1 = IntentMapper.get_fallback_response(intent)
        response_2 = IntentMapper.get_fallback_response(intent)

        # Should be identical, not randomized
        assert response_1 == response_2
        assert isinstance(response_1, str)
        assert len(response_1) > 0


class TestDeterminismViolationDetection:
    """Tests to catch any determinism violations"""

    def test_no_random_numbers_in_intent_extraction(self):
        """Verify no randomness in intent extraction"""
        messages = [
            "add task 1",
            "list tasks",
            "complete task 1",
            "delete task 1",
        ]

        for message in messages:
            results = [
                IntentMapper.extract_intent(message) for _ in range(10)
            ]

            # All results should be identical
            for result in results[1:]:
                assert result == results[0]

    def test_no_timestamp_dependencies_in_intents(self):
        """Verify intent extraction doesn't depend on time"""
        import time

        message = "add task"

        intent_1 = IntentMapper.extract_intent(message)[0]
        time.sleep(0.1)  # Wait 100ms
        intent_2 = IntentMapper.extract_intent(message)[0]

        assert intent_1 == intent_2

    def test_no_global_state_affecting_intents(self):
        """Verify no global state affects intent determination"""
        message = "update task 1 to new title"

        # Extract intent, modify nothing, extract again
        intent_1, conf_1 = IntentMapper.extract_intent(message)

        # Process different message
        IntentMapper.extract_intent("list tasks")
        IntentMapper.extract_intent("add something")

        # Original should still be the same
        intent_2, conf_2 = IntentMapper.extract_intent(message)

        assert (intent_1, conf_1) == (intent_2, conf_2)
