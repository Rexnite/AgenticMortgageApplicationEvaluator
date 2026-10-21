from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    MISSING_INPUT = "MISSING_INPUT"
    EXTRACTION_FAILURE = "EXTRACTION_FAILURE"
    LLM_FAILURE = "LLM_FAILURE"
    SCHEMA_VALIDATION_FAILURE = "SCHEMA_VALIDATION_FAILURE"
    POLICY_FAILURE = "POLICY_FAILURE"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"


class WorkflowError(BaseModel):
    stage: str
    agent: str | None = None
    code: str
    message: str
    recoverable: bool = False
    category: ErrorCategory = ErrorCategory.INFRASTRUCTURE_FAILURE


class ReviewStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED_BY_REVIEWER = "APPROVED_BY_REVIEWER"
    DECLINED_BY_REVIEWER = "DECLINED_BY_REVIEWER"
    REQUEST_MORE_INFORMATION = "REQUEST_MORE_INFORMATION"


class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    DECLINE = "DECLINE"
    REQUEST_MORE_INFORMATION = "REQUEST_MORE_INFORMATION"


class WorkflowStatus(str, Enum):
    RUNNING = "RUNNING"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    AWAITING_INFORMATION = "AWAITING_INFORMATION"
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"


class HumanReviewDecision(BaseModel):
    decision: ReviewDecision
    reviewer_comment: str | None = None


class AuditEvent(BaseModel):
    timestamp: str
    application_id: str
    event_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanReview(BaseModel):
    reason: str
    uncertain_checks: list[str] = Field(default_factory=list)
    relevant_evidence: list[str] = Field(default_factory=list)
    policy_rules: list[str] = Field(default_factory=list)
    recommended_action: str = "Review evidence and record a decision"
    comments: str | None = None
    status: ReviewStatus = ReviewStatus.PENDING_REVIEW


class ReflectionResult(BaseModel):
    passed: bool = True
    issues: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    policy_inconsistencies: list[str] = Field(default_factory=list)
    confidence_concerns: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
