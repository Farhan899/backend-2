"""
Unit Tests for Intent Mapper

Tests deterministic intent extraction and mapping per Constitutional Section VI.
"""

import pytest
from app.services.intent_mapping import IntentMapper, Intent


class TestIntentExtraction:
    """Tests for intent extraction"""

    def test_add_intent_detection(self):
        """Test detection of add intent"""
        test_cases = [
            "add buy groceries",
            "create a new task",
            "new task: remember milk",
            "remember to call mom",
            "i need to finish homework",
        ]

        for message in test_cases:
            intent, confidence = IntentMapper.extract_intent(message)
            assert intent == Intent.ADD, f"Failed for message: {message}"
            assert confidence >= 0.7, f"Low confidence for: {message}"

    def test_list_intent_detection(self):
        """Test detection of list intent"""
        test_cases = [
            "list my tasks",
            "show all tasks",
            "what tasks do i have",
            "my tasks",
            "show my tasks",
        ]

        for message in test_cases:
            intent, confidence = IntentMapper.extract_intent(message)
            assert intent == Intent.LIST, f"Failed for message: {message}"
            assert confidence >= 0.7

    def test_complete_intent_detection(self):
        """Test detection of complete intent"""
        test_cases = [
            "complete task 1",
            "mark task 1 done",
            "done with task 1",
            "finish task 1",
            "check off task 1",
        ]

        for message in test_cases:
            intent, confidence = IntentMapper.extract_intent(message)
            assert intent == Intent.COMPLETE, f"Failed for message: {message}"
            assert confidence >= 0.7

    def test_delete_intent_detection(self):
        """Test detection of delete intent"""
        test_cases = [
            "delete task 1",
            "remove task 1",
            "trash task 1",
            "get rid of task 1",
        ]

        for message in test_cases:
            intent, confidence = IntentMapper.extract_intent(message)
            assert intent == Intent.DELETE, f"Failed for message: {message}"
            assert confidence >= 0.7

    def test_update_intent_detection(self):
        """Test detection of update intent"""
        test_cases = [
            "update task 1",
            "edit task 1",
            "change task 1",
            "modify task 1",
        ]

        for message in test_cases:
            intent, confidence = IntentMapper.extract_intent(message)
            assert intent == Intent.UPDATE, f"Failed for message: {message}"
            assert confidence >= 0.7

    def test_unknown_intent_detection(self):
        """Test detection of unknown intent"""
        test_cases = [
            "hello",
            "what's the weather",
            "tell me a joke",
            "how are you",
        ]

        for message in test_cases:
            intent, confidence = IntentMapper.extract_intent(message)
            assert intent == Intent.UNKNOWN, f"Should be unknown: {message}"
            assert confidence < 0.5

    def test_deterministic_intent_mapping(self):
        """Test that identical intents always produce same result"""
        message = "add buy groceries"

        results = []
        for _ in range(5):
            intent, confidence = IntentMapper.extract_intent(message)
            results.append((intent, confidence))

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]

    def test_case_insensitive_intent_detection(self):
        """Test that intent detection is case-insensitive"""
        test_pairs = [
            ("add task", "ADD TASK"),
            ("list tasks", "LIST TASKS"),
            ("complete task", "COMPLETE TASK"),
        ]

        for lower, upper in test_pairs:
            lower_intent, _ = IntentMapper.extract_intent(lower)
            upper_intent, _ = IntentMapper.extract_intent(upper)
            assert lower_intent == upper_intent


class TestIntentUtilities:
    """Tests for intent utility functions"""

    def test_get_tool_name(self):
        """Test getting MCP tool name from intent"""
        assert IntentMapper.get_tool_name(Intent.ADD) == "add_task"
        assert IntentMapper.get_tool_name(Intent.LIST) == "list_tasks"
        assert IntentMapper.get_tool_name(Intent.GET) == "get_task"
        assert IntentMapper.get_tool_name(Intent.UPDATE) == "update_task"
        assert IntentMapper.get_tool_name(Intent.DELETE) == "delete_task"
        assert IntentMapper.get_tool_name(Intent.COMPLETE) == "complete_task"

    def test_should_confirm_delete(self):
        """Test that delete operations require confirmation"""
        assert IntentMapper.should_confirm(Intent.DELETE) is True
        assert IntentMapper.should_confirm(Intent.ADD) is False
        assert IntentMapper.should_confirm(Intent.UPDATE) is False

    def test_fallback_responses(self):
        """Test fallback responses for each intent"""
        for intent in Intent:
            response = IntentMapper.get_fallback_response(intent)
            assert isinstance(response, str)
            assert len(response) > 0
