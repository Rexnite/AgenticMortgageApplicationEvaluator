from pathlib import Path
import io
import logging

import pymupdf as fitz
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from models.extraction import ExtractionResult


class OCRExtractor:

    _engine = RapidOCR()

    @classmethod
    def extract(cls, file_path: Path):

        doc = fitz.open(file_path)

        extracted_pages = []

        logger = logging.getLogger("mortgage.ocr")
        logger.info("ocr_started", extra={"page_count": len(doc)})

        for page_number, page in enumerate(doc):

            logger.debug("ocr_page_started", extra={"page_number": page_number + 1})

            pix = page.get_pixmap(dpi=300)

            image = Image.open(
                io.BytesIO(
                    pix.tobytes("png")
                )
            )

            result, _ = cls._engine(np.array(image))

            page_text = ""

            if result:
                for line in result:
                    page_text += line[1] + "\n"

            extracted_pages.append(page_text)

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
