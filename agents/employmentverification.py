from agents.base_agent import BaseAgent
from models.employmentverification import EmploymentVerification
from services.comparisons import all_normalized_match



class EmploymentVerificationAgent(BaseAgent):
    name = "employment_verification"
    input_fields = ("employment", "payslips")
    output_fields = ("employmentverification",)

    @staticmethod
    def deterministic_result(state) -> EmploymentVerification | None:
        employer = state.employment.employer_name
        employer_match = all_normalized_match([employer, *(p.employer_name for p in state.payslips)])
        names = [p.employee_name for p in state.payslips]
        applicant = next((name for name in names if name), None)
        if not employer or not applicant or employer_match is not True:
            return None
        return EmploymentVerification(applicant_name=applicant, verified=True,
            employer=employer, inconsistencies=[], missing_information=[], confidence=1.0)

    def extract(self, state) -> EmploymentVerification:

        prompt = f"""

            Employment
            {state.employment.model_dump_json(indent=2)}

            Payslips
            {[p.model_dump() for p in state.payslips]}

            Compare all documents.
            
            Check the following and if they match across the documents
            
            Employer

            Job Title

            Employment Type

            Start Date

            Salary

            Probation

            Length of employment

            Determine:

            • Applicant name

            • Employment verified?

            • Missing information

            • Inconsistencies

            Return ONLY valid JSON.

            The JSON MUST EXACTLY match this schema.

            {{
                "applicant_name": "",
                "verified": true,
                "employer": "AstraZeneca",
                "missing_information": [
                    ""
                ],

                "inconsistencies": [
                    ""
                ],

                "confidence": 0.95
            }}

            Rules:

            - missing_information MUST be a flat array of strings.
            - Do NOT group missing information by document.
            - confidence is REQUIRED.
            - Do NOT return any additional fields.
            - Do NOT return "details".
            """

        result = self.chat_json(
            "",
            prompt, EmploymentVerification
        )
        
        # Add confidence if missing
        result.setdefault("confidence", 0.9)
        
        employment_verification = EmploymentVerification(**result)

        return employment_verification
    
    
    def run(self, state):

        state.current_step = "Employment Verification"

        state.employmentverification = self.deterministic_result(state) or self.extract(state)

        state.completed_steps.append("Employment Verification")

        return state
