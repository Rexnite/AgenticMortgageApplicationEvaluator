from pydantic import BaseModel


class ExtractionResult(BaseModel):

    text: str

    extraction_method: str

    pages: int

    success: bool

    confidence: float = 1.0