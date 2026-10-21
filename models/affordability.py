from pydantic import BaseModel



class AffordabilityData(BaseModel):

    annual_salary: float | None = None

    average_monthly_gross_income: float | None = None
    average_monthly_net_income: float | None = None

    average_monthly_expenses: float | None = None

    disposable_income: float | None = None

    debt_to_income_ratio: float | None = None

    affordability_score: int | None = None

    recommendation: str | None = None

    confidence: float = 0.0