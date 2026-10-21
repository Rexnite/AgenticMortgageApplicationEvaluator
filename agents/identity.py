from agents.base_agent import BaseAgent
from models.identity import IdentityVerification
from services.comparisons import all_normalized_match



class IdentityVerificationAgent(BaseAgent):
    name = "identity_verification"
    input_fields = ("passport", "payslips", "bank_statements")
    output_fields = ("identity",)

    @staticmethod
    def deterministic_result(state) -> IdentityVerification | None:
        applicant = " ".join(filter(None, (state.passport.given_names, state.passport.surname))).strip()
        observed = [applicant]
        observed.extend(p.employee_name for p in state.payslips)
        observed.extend(b.account_holder_name for b in state.bank_statements)
        matches = all_normalized_match(observed)
        if not applicant or matches is None or not matches:
            return None
        missing = [] if state.passport.extraction_complete else ["Passport extraction is incomplete"]
        return IdentityVerification(applicant_name=applicant, verified=not missing,
            inconsistencies=[], missing_information=missing, confidence=1.0 if not missing else 0.8)

    def extract(self, state) -> IdentityVerification:
        
        
        

        prompt = f"""

            Passport
            {state.passport.model_dump_json(indent=2)}

            Payslips
            {[p.model_dump() for p in state.payslips]}

            Bank Statements
            {[b.model_dump() for b in state.bank_statements]}

            Compare all documents.
            
            Check
            Name matches

            DOB matches

            Passport complete

            Passport expiry

            Nationality

            Address (if available)

            Return ONLY valid JSON.

            The JSON MUST EXACTLY match this schema.

            {{
                "applicant_name": "",
                "verified": true,
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
            prompt, IdentityVerification
        )
        
        # Add confidence if missing
        result.setdefault("confidence", 0.9)
        
        identity = IdentityVerification(**result)

        return identity
    
    
    def run(self, state):

        state.current_step = "Identity Verification"

        state.identity = self.deterministic_result(state) or self.extract(state)

        state.completed_steps.append("Identity Verification")

        return state
