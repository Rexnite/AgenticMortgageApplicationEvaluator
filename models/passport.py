from pydantic import BaseModel


class PassportData(BaseModel):

    passport_number: str | None = None
    surname: str | None = None
    given_names: str | None = None
    nationality: str | None = None
    date_of_birth: str | None = None
    age: float | None = None
    sex: str | None = None
    place_of_birth: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None
    issuing_country: str | None = None

    confidence: float = 0.0

    extraction_complete: bool = False
