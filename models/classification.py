from pydantic import BaseModel


class DocumentClassification(BaseModel):

    document_type: str

    confidence: float

    reason: str