from pydantic import BaseModel


class IncomeVerification(BaseModel):

    applicant_name: str
    verified: bool
    monthly_income: float
    salary_credits: float
    inconsistencies: list[str]
    missing_information: list[str]
    confidence: float