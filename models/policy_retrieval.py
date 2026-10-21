from typing import Any

from pydantic import BaseModel, Field


class PolicyDocument(BaseModel):
    """Retrieved supporting context with explicit provenance."""

    id: str
    title: str
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
