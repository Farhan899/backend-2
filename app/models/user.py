from sqlmodel import Field, SQLModel
from datetime import datetime
from typing import Optional

class User(SQLModel, table=True):
    """User entity from Better Auth"""
    __tablename__ = "user"

    id: str = Field(primary_key=True)
    email: str = Field(unique=True, index=True)
    emailVerified: bool = Field(default=False, sa_column_kwargs={"name": "emailVerified"})
    name: Optional[str] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"name": "createdAt"})
    updatedAt: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"name": "updatedAt"})
    image: Optional[str] = None
