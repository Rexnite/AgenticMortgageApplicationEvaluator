from pathlib import Path
import logging

from tools.extraction.pdf_text_extractor import PDFTextExtractor
from tools.extraction.ocr_extractor import OCRExtractor


class DocumentExtractionTool:

    @staticmethod
    def extract(file_path: Path):

        result = PDFTextExtractor.extract(file_path)

        if result.success:
            logging.getLogger("mortgage.extraction").info("pdf_text_extraction_completed")
            return result

        logging.getLogger("mortgage.extraction").info("ocr_fallback_started")
        
        ocr_result = OCRExtractor.extract(file_path)

        logging.getLogger("mortgage.extraction").info("ocr_fallback_completed", extra={"character_count": len(ocr_result.text)})

        return ocr_result
