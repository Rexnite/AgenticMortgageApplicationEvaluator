from agents.base_agent import BaseAgent
from models.income import IncomeVerification
from services.calculations import average_payslip_income, compare_income_sources, recurring_salary_credit



class IncomeVerificationAgent(BaseAgent):
    name = "income_verification"
    input_fields = ("employment", "payslips", "bank_statements")
    output_fields = ("income",)

    @staticmethod
    def deterministic_result(state) -> IncomeVerification | None:
        monthly_income = average_payslip_income(state.payslips)
        salary_credits = recurring_salary_credit(state.bank_statements)
        consistent = compare_income_sources(monthly_income, salary_credits)
        applicant = next((p.employee_name for p in state.payslips if p.employee_name), None)
        if not applicant or consistent is not True:
            return None
        return IncomeVerification(applicant_name=applicant, verified=True,
            monthly_income=monthly_income, salary_credits=salary_credits,
            inconsistencies=[], missing_information=[], confidence=1.0)

    def extract(self, state) -> IncomeVerification:

        prompt = f"""

            Employment
            {state.employment.model_dump_json(indent=2)}

            Payslips
            {[p.model_dump() for p in state.payslips]}

            Bank Statements
            {[b.model_dump() for b in state.bank_statements]}

            Compare all documents.
            
            Check the following and if they match across the documents
            
            Salary consistency

            Gross Pay

            Net Pay

            Salary deposits

            Salary frequency

            Missing months

            Bonuses

            Overtime

            Determine:

            • Applicant name

            • Income verified?

            • Missing information

            • Inconsistencies

            Return ONLY valid JSON.

            The JSON MUST EXACTLY match this schema.

            {{
                "applicant_name": "",
                "verified": true,
                "monthly_income": float,
                "salary_credits": float,

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
            prompt, IncomeVerification
        )
        
        # Add confidence if missing
        result.setdefault("confidence", 0.9)
        #result["total_income"] = to_float(result.get("total_income"))
        
        income = IncomeVerification(**result)

        return income
    
    
    def run(self, state):

        state.current_step = "Income Verification"

        state.income = self.deterministic_result(state) or self.extract(state)

        state.completed_steps.append("Income Verification")

        return state
