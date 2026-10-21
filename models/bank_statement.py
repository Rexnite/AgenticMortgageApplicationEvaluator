from pydantic import BaseModel, Field

class BankStatementData(BaseModel):

    bank_name: str | None = None
    account_holder_name: str | None = None
    account_number: str | None = None
    sort_code: str | None = None

    statement_period: str | None = None
    statement_months: float | None = None

    opening_balance: float | None = None
    closing_balance: float | None = None

    total_income: float | None = None
    total_expenses: float | None = None

    salary_credits: list[float] = Field(default_factory=list)

    overdraft_limit: float | None = None

    confidence: float = 0.0
    extraction_complete: bool = False
