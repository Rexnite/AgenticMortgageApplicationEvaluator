from agents.base_agent import BaseAgent
from models.bank import BankVerification
from services.comparisons import all_normalized_match



class BankVerificationAgent(BaseAgent):
    name = "bank_verification"
    input_fields = ("bank_statements",)
    output_fields = ("bank",)

    @staticmethod
    def deterministic_result(state) -> BankVerification | None:
        statements = state.bank_statements
        if not statements:
            return None
        holders_match = all_normalized_match(b.account_holder_name for b in statements) if len(statements) > 1 else bool(statements[0].account_holder_name)
        accounts_match = all_normalized_match(b.account_number for b in statements) if len(statements) > 1 else bool(statements[0].account_number)
        sort_codes_match = all_normalized_match(b.sort_code for b in statements) if len(statements) > 1 else bool(statements[0].sort_code)
        months = [b.statement_months for b in statements]
        if holders_match is not True or accounts_match is not True or sort_codes_match is not True or any(value is None for value in months):
            return None
        return BankVerification(applicant_name=statements[0].account_holder_name,
            verified=True, statement_months=round(sum(months)),
            overdraft=any((b.overdraft_limit or 0) > 0 for b in statements),
            inconsistencies=[], missing_information=[], confidence=1.0)

    def extract(self, state) -> BankVerification:

        prompt = f"""

            Bank Statements
            {[b.model_dump() for b in state.bank_statements]}

            Compare all documents.
            
            Check the following
            
            Account holder

            Statement months

            Account number

            Salary deposits

            Overdraft

            Returned payments

            Large cash withdrawals

            Determine:
            
            • Applicant name

            • Bank account verified?

            • Missing information

            • Inconsistencies

            Return ONLY valid JSON.

            The JSON MUST EXACTLY match this schema.

            {{
                "applicant_name": "",
                "verified": True,
                "statement_months": 3,
                "overdraft": True,

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
            prompt, BankVerification
        )
        
        # Add confidence if missing
        result.setdefault("confidence", 0.9)
        
        bank = BankVerification(**result)

        return bank
    
    
    def run(self, state):

        state.current_step = "Bank Account Verification"

        state.bank = self.deterministic_result(state) or self.extract(state)

        state.completed_steps.append("Bank Account Verification")

        return state
