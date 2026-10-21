from pydantic import BaseModel


class IdentityVerification(BaseModel):

    applicant_name: str

    verified: bool

    inconsistencies: list[str]

    missing_information: list[str]

    confidence: float