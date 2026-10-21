from pydantic import BaseModel, Field

from models.document import Document
from models.passport import PassportData


class MortgageApplication(BaseModel):

    application_id: str

    documents: list[Document] = Field(default_factory=list)

    passport: PassportData | None = None

    payslips: list = Field(default_factory=list)

    bank_statements: list = Field(default_factory=list)

    employment: dict | None = None

    affordability: dict | None = None

    underwriting_decision: dict | None = None