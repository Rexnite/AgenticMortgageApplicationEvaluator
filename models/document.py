from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:

    file_name: str

    file_path: Path

    extension: str

    size: int

    document_type: str = "Unknown"

    extracted_text: str = ""

    confidence: float = 0.0
    
    classification_reason: str = ""
        
    extraction_method: str = ""

    extraction_success: bool | None = None

    extraction_confidence: float | None = None

    page_count: int | None = None
