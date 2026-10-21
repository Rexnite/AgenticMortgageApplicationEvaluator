from pydantic import BaseModel

class DocumentOwnership(BaseModel):
    document_type: str
    document_index: int

    owner_name: str

    belongs_to_applicant: bool

    confidence: float

    reason: str