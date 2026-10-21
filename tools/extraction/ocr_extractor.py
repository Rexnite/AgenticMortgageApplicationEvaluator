from pathlib import Path
import logging

import pymupdf as fitz

from models.extraction import ExtractionResult


class OCRExtractor:

    @classmethod
    def extract(cls, file_path: Path):

        doc = fitz.open(file_path)

        extracted_pages = []

        logger = logging.getLogger("mortgage.ocr")
        logger.info("ocr_started", extra={"page_count": len(doc)})

        for page_number, page in enumerate(doc):

            logger.debug("ocr_page_started", extra={"page_number": page_number + 1})

            try:
                text_page = page.get_textpage_ocr(dpi=300, full=True)
                extracted_pages.append(page.get_text(textpage=text_page))
            except RuntimeError as exc:
                logger.warning("ocr_page_failed", extra={
                    "page_number": page_number + 1,
                    "error_kind": type(exc).__name__,
                })
                raise RuntimeError(
                    "OCR_UNAVAILABLE: Tesseract OCR is not installed or could not process the document"
                ) from exc

        doc.close()

        text = "\n".join(extracted_pages)

        logger.info("ocr_completed", extra={"character_count": len(text)})

        return ExtractionResult(
            text=text,
            extraction_method="ocr",
            pages=len(extracted_pages),
            success=len(text) > 20,
            confidence=0.90,
        )
