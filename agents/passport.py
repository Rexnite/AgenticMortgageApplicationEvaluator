from agents.base_agent import BaseAgent
from models.passport import PassportData
from core.prompts import PASSPORT_EXTRACTION_SYSTEM
from services.calculations import calculate_age


class PassportAgent(BaseAgent):
    name = "passport_extraction"
    input_fields = ("documents",)
    output_fields = ("passport",)

    def extract(self, text: str) -> PassportData:

        prompt = f"""
            Extract the following:

            - Passport Number
            - Surname
            - Given Names
            - Nationality
            - Date of Birth
            - Age
            - Place of Birth
            - Sex
            - Issue Date
            - Expiry Date
            - Issuing Country

            Passport OCR Text:

            {text}
            """

        result = self.chat_json(
            PASSPORT_EXTRACTION_SYSTEM,
            prompt, PassportData
        )
        passport = PassportData(**result)
        calculated_age = calculate_age(passport.date_of_birth)
        if calculated_age is not None:
            passport.age = calculated_age
        required = [
        passport.passport_number,
        passport.surname,
        passport.given_names,
        passport.date_of_birth,
        passport.age,
        passport.expiry_date,
        ]

        passport.extraction_complete = all(required)

        return passport
    
    
    def run(self, state):

        state.current_step = "Passport Extraction"

        for document in state.documents:

            if document.document_type != "Passport":
                continue

            state.passport = self.extract(
                document.extracted_text
            )

        state.completed_steps.append(
            "Passport Extraction"
        )

        return state
