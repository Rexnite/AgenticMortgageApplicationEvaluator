from pydantic import BaseModel, Field
from enum import Enum


class Operator(str, Enum):
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_THAN_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"



class PolicyRule(BaseModel):
    rule_id: str
    name: str
    description: str

    state_path: str
    operator: Operator
    expected_value: object
    
    severity: str = "Mandatory"   # Mandatory | Warning | Advisory
    on_fail: str = "FAIL"      # FAIL / REFER
    enabled: bool = True
    mandatory: bool = True
        
        
class PolicyResult(BaseModel):
    rule_id: str
    name: str
    actual_value: object
    expected_value: object
    status: str      # PASS | FAIL | REFER
    reason: str
    mandatory: bool = True


class PolicyAssessment(BaseModel):
    passed: int = 0
    failed: int = 0
    referred: int = 0
    overall_decision: str = "PASS"
    rules: list[PolicyResult] = Field(default_factory=list)
    confidence: float = 1.0
