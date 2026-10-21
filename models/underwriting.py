from pydantic import BaseModel, Field

class UnderwritingDecision(BaseModel):

    decision: str          # Approve / Refer / Decline

    confidence: float | None = None

    risk_rating: str       # Low / Medium / High

    strengths: list[str] = Field(default_factory=list)

    concerns: list[str] = Field(default_factory=list)

    conditions: list[str] = Field(default_factory=list)

    summary: str | None = None
