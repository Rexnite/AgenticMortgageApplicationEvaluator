from enum import Enum

from pydantic import BaseModel, Field


class FraudSignalStatus(str, Enum):
    DETECTED = "DETECTED"
    SUSPECTED = "SUSPECTED"
    NOT_DETECTED = "NOT_DETECTED"
    NOT_ASSESSED = "NOT_ASSESSED"


class FraudSignalCategory(str, Enum):
    IDENTITY = "IDENTITY"
    EMPLOYMENT = "EMPLOYMENT"
    INCOME = "INCOME"
    BANKING = "BANKING"
    DOCUMENT_INTEGRITY = "DOCUMENT_INTEGRITY"
    CROSS_DOCUMENT = "CROSS_DOCUMENT"


class FraudSignalSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FraudSignal(BaseModel):
    code: str
    category: FraudSignalCategory
    status: FraudSignalStatus
    severity: FraudSignalSeverity
    confidence: float | None = None
    description: str
    evidence_summary: str | None = None


class FraudAssessment(BaseModel):

    applicant_name: str
    fraud_score: int
    risk: str       # Low / Medium / High
    indicators: list[str] = Field(default_factory=list)
    confidence: float
    signals: list[FraudSignal] = Field(default_factory=list)
    confirmed_inconsistencies: list[str] = Field(default_factory=list)
    suspected_concerns: list[str] = Field(default_factory=list)
    unassessed_checks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_action: str | None = None


class FraudSynthesis(BaseModel):
    suspected_concerns: list[str] = Field(default_factory=list)
    recommended_action: str = ""
    confidence: float = 0.0
