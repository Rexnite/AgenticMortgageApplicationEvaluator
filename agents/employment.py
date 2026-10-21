from agents.base_agent import BaseAgent
from models.employment import EmploymentData
from core.prompts import EMPLOYMENT_EXTRACTION_SYSTEM


class EmploymentAgent(BaseAgent):
    name = "employment_extraction"
    input_fields = ("documents",)
    output_fields = ("employment",)

    def extract(self, text: str) -> EmploymentData:

        prompt = f"""
            Extract the following:

            - Employer Name
            - Job Title
            - Employment Type
            - Annual Salary
            - Weekly Hours
            - Start Date
            - Probation


            Employment OCR Text:

            {text}
            """

        result = self.chat_json(
            EMPLOYMENT_EXTRACTION_SYSTEM,
            prompt, EmploymentData
        )
        employment = EmploymentData(**result)
        required = [
        employment.employer_name,
        employment.job_title,
        employment.annual_salary,
        employment.start_date,
        employment.probation,
        ]

        employment.extraction_complete = all(required)

        return employment
    
    
    def run(self, state):

        state.current_step = "Employment Extraction"

        for document in state.documents:

            if document.document_type == "Employment Letter":
                state.employment = self.extract(
                    document.extracted_text
                )

        state.completed_steps.append(
            "Employment Extraction"
        )

        return state
