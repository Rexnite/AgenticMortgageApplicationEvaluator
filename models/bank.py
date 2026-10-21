from pydantic import BaseModel


class BankVerification(BaseModel):

    applicant_name: str

    verified: bool
        
    statement_months: int
        
    overdraft: bool

    inconsistencies: list[str]

    missing_information: list[str]

    confidence: float