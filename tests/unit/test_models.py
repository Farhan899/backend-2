import pytest
from datetime import datetime
from app.models.task import Task

def test_task_model_creation():
    """Test Task model instantiation"""
    task = Task(
        user_id="usr_123",
        title="Test Task",
        description="Test description",
    )
    assert task.user_id == "usr_123"
    assert task.title == "Test Task"
    assert task.description == "Test description"
    assert task.is_completed is False
    assert isinstance(task.created_at, datetime)
    assert isinstance(task.updated_at, datetime)

def test_task_model_without_description():
    """Test Task model with optional description"""
    task = Task(user_id="usr_123", title="Test Task")
    assert task.description is None

def test_task_model_has_required_fields():
    """Test that Task model has required fields defined"""
    # Verify model has the expected fields
    assert hasattr(Task, 'title')
    assert hasattr(Task, 'user_id')
    # Constraints are checked at API layer with Pydantic schemas

def test_task_model_field_defaults():
    """Test Task model field defaults"""
    task = Task(user_id="usr_123", title="Test")
    assert task.is_completed is False
    assert task.description is None
    assert isinstance(task.created_at, datetime)
    assert isinstance(task.updated_at, datetime)

def test_task_model_tablename():
    """Test that table name is correctly set"""
    assert Task.__tablename__ == "tasks"

def test_task_model_primary_key():
    """Test that id is the primary key"""
    task = Task(user_id="usr_123", title="Test")
    # id should be None before database insert
    assert task.id is None
