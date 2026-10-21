from enum import Enum

from pydantic import BaseModel


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REFER = "REFER"
    NOT_ASSESSED = "NOT_ASSESSED"


class CheckResult(BaseModel):
    stage: str
    check_code: str
    check_name: str
    status: CheckStatus
    reason: str
    document_type: str | None = None
    metric: str | None = None
    source: str | None = None
    action: str | None = None
