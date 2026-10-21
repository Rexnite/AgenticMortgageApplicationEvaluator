from pydantic import BaseModel


class EmploymentVerification(BaseModel):

    applicant_name: str
    verified: bool
    employer: str
    inconsistencies: list[str]
    missing_information: list[str]

    confidence: float