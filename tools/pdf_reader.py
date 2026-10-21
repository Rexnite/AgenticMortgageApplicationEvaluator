import pymupdf as fitz
from pathlib import Path


class PDFReader:

    @staticmethod
    def extract_text(file_path: Path) -> str:
        """Extract text from a PDF document."""

        document = fitz.open(file_path)

        pages = []

        for page in document:
            pages.append(page.get_text())

        document.close()

        return "\n".join(pages).strip()
    
    @staticmethod
    def has_text(text):

        return len(text.strip()) > 50
