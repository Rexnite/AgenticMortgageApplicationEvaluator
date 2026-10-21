from pydantic import BaseModel

class PolicyAssessmentSummary(BaseModel):

    total_rules: int 

    passed_rules: int

    failed_rules: int

    referred_rules: int

    skipped_rules: int = 0

    overall_status: str
    # PASS
    # REFER
    # FAIL

    mandatory_failures: list[str]

    referred_rules_list: list[str]

    failed_rules_list: list[str]

    passed_rules_list: list[str]

    summary: str

    confidence: float