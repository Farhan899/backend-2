from typing import List, Annotated
from datetime import datetime
from fastapi import APIRouter, Depends, Path, HTTPException, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.task import Task
from app.schemas.task import TaskCreate, TaskUpdate, TaskResponse
from app.api.deps import get_current_user
from app.core.database import get_session
from app.core.security import verify_user_access

router = APIRouter(prefix="/api/{user_id}", tags=["tasks"])

@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    user_id: Annotated[str, Path()],
    current_user: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> List[Task]:
    """
    List all tasks for the authenticated user.

    Returns tasks ordered by creation date (most recent first).
    """
    # Verify user can access this resource
    verify_user_access(token_user_id=current_user, path_user_id=user_id)

    # Query tasks filtered by user_id
    statement = (
        select(Task)
        .where(Task.user_id == user_id)
        .order_by(Task.created_at.desc())
    )
    result = await session.execute(statement)
    tasks = result.scalars().all()

    return list(tasks)

@router.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    user_id: Annotated[str, Path()],
    task_data: TaskCreate,
    current_user: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Task:
    """
    Create a new task for the authenticated user.
    """
    # Verify user can access this resource
    verify_user_access(token_user_id=current_user, path_user_id=user_id)

    # Create task with validated data
    task = Task(
        user_id=user_id,
        title=task_data.title,
        description=task_data.description,
        priority=task_data.priority,
        due_date=task_data.due_date,
    )

    session.add(task)
    await session.commit()
    await session.refresh(task)

    return task

@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    user_id: Annotated[str, Path()],
    task_id: Annotated[int, Path()],
    current_user: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Task:
    """
    Get a specific task by ID. User must own the task.
    """
    # Verify user can access this resource
    verify_user_access(token_user_id=current_user, path_user_id=user_id)

    # Fetch task
    statement = select(Task).where(Task.id == task_id, Task.user_id == user_id)
    result = await session.execute(statement)
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task

@router.put("/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    user_id: Annotated[str, Path()],
    task_id: Annotated[int, Path()],
    task_data: TaskUpdate,
    current_user: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Task:
    """
    Update an existing task.
    """
    verify_user_access(token_user_id=current_user, path_user_id=user_id)

    # Fetch task
    statement = select(Task).where(Task.id == task_id, Task.user_id == user_id)
    result = await session.execute(statement)
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Update fields if provided
    if task_data.title is not None:
        task.title = task_data.title
    if task_data.description is not None:
        task.description = task_data.description
    if task_data.priority is not None:
        task.priority = task_data.priority
    if task_data.due_date is not None:
        task.due_date = task_data.due_date

    task.updated_at = datetime.utcnow()

    session.add(task)
    await session.commit()
    await session.refresh(task)

    return task

@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    user_id: Annotated[str, Path()],
    task_id: Annotated[int, Path()],
    current_user: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """
    Delete a task permanently.
    """
    verify_user_access(token_user_id=current_user, path_user_id=user_id)

    # Fetch task
    statement = select(Task).where(Task.id == task_id, Task.user_id == user_id)
    result = await session.execute(statement)
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    await session.delete(task)
    await session.commit()

@router.patch("/tasks/{task_id}/complete", response_model=TaskResponse)
async def toggle_task_completion(
    user_id: Annotated[str, Path()],
    task_id: Annotated[int, Path()],
    current_user: Annotated[str, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Task:
    """
    Toggle task completion status (complete ↔ incomplete).
    """
    verify_user_access(token_user_id=current_user, path_user_id=user_id)

    # Fetch task
    statement = select(Task).where(Task.id == task_id, Task.user_id == user_id)
    result = await session.execute(statement)
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Toggle completion status
    task.is_completed = not task.is_completed
    task.updated_at = datetime.utcnow()

    session.add(task)
    await session.commit()
    await session.refresh(task)

    return task
