import pymupdf as fitz
from pathlib import Path

from models.extraction import ExtractionResult


class PDFTextExtractor:

    @staticmethod
    def extract(file_path: Path) -> ExtractionResult:

        document = fitz.open(file_path)

        pages = []
        page_count = len(document)

        for page in document:
            pages.append(page.get_text())

        document.close()

        text = "\n".join(pages).strip()

        return ExtractionResult(
            text=text,
            extraction_method="pdf_text",
            pages=page_count,
            success=(len(text.strip()) > 50 and any(ch.isalpha() for ch in text)),
            confidence=1.0,
        )
