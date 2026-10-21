from pathlib import Path
import logging

from agents.base_agent import BaseAgent
from models.document import Document
from tools.file_reader import FileReader
from models.classification import DocumentClassification
from core.prompts import DOCUMENT_CLASSIFIER_SYSTEM
from tools.extraction.extraction_tool import DocumentExtractionTool


class DocumentAgent(BaseAgent):
    name = "document_intake"

    output_fields = ("documents",)

    def __init__(self, llm=None):
        super().__init__(llm=llm)

    def run(self, state, folder):

        state.current_step = "Document Scan"
        state.documents = self.scan(folder)
        state.completed_steps.append("Document Scan")

        return state
    
    def scan(self, folder: Path) -> list[Document]:

        logger = logging.getLogger("mortgage.document")

        files = FileReader.get_documents(folder)

        documents = []

        for file in files:

            document = Document(
                file_name=file.name,
                file_path=file,
                extension=file.suffix,
                size=file.stat().st_size,
            )

            document = self.classify_file(document.file_path)

            documents.append(document)

            logger.info("document_classified", extra={"document_type": document.document_type})
            
            #if document.document_type == "Passport":

                #passport = self.passport_agent.extract(document.extracted_text)

                #print(passport)

        return documents


    
    def classify_file(self, file_path: Path) -> Document:

        document = Document(
        file_name=file_path.name,
        file_path=file_path,
        extension=file_path.suffix,
        size=file_path.stat().st_size,
        )
        extraction = DocumentExtractionTool.extract(file_path)
        #extraction.texttext = PDFReader.extract_text(file_path)
        document.extracted_text = extraction.text
        document.extraction_method = extraction.extraction_method
        document.extraction_success = extraction.success
        document.extraction_confidence = extraction.confidence
        document.page_count = extraction.pages
        prompt = f"""
            Document Text:
            {document.extracted_text[:5000]}
            """

        result = self.chat_json(
            DOCUMENT_CLASSIFIER_SYSTEM,
            prompt, DocumentClassification
        )

        classification = DocumentClassification(**result)
        document.document_type = classification.document_type
        document.confidence = classification.confidence
        document.classification_reason = classification.reason

        return document
