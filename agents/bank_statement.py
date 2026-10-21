from agents.base_agent import BaseAgent
from models.bank_statement import BankStatementData
from core.prompts import BANK_STATEMENT_EXTRACTION_SYSTEM
from utils.parser import to_float
from pydantic import BaseModel, Field


class BankStatementExtraction(BaseModel):
    """Raw local-model fields; money may be formatted text before normalization."""

    bank_name: str | None = None
    account_holder_name: str | None = None
    account_number: str | None = None
    sort_code: str | None = None
    statement_period: str | None = None
    statement_months: str | float | int | None = None
    opening_balance: str | float | int | None = None
    closing_balance: str | float | int | None = None
    total_income: str | float | int | None = None
    total_expenses: str | float | int | None = None
    salary_credits: list[str | float | int] = Field(default_factory=list)
    overdraft_limit: str | float | int | None = None
    confidence: float = 0.0



class BankStatementAgent(BaseAgent):
    name = "bank_statement_extraction"
    input_fields = ("documents",)
    output_fields = ("bank_statements",)

    def extract(self, text: str) -> BankStatementData:

        prompt = f"""
            Extract the following:
            
            - Bank Name
            - Account Holder
            - Account Number
            - Sort Code
            - Statement Period
            - Statement Months
            - Opening Balance
            - Closing Balance
            - Total Income
            - Total Expenses
            - Salary Credits (list)
            - Overdraft Limit

            Bank Statement OCR Text:

            {text}
            """
        
        
        result = self.chat_json(
            BANK_STATEMENT_EXTRACTION_SYSTEM,
            prompt, BankStatementExtraction
        )
        
        result["statement_months"] = to_float(result.get("statement_months"))
        result["opening_balance"] = to_float(result.get("opening_balance"))
        result["closing_balance"] = to_float(result.get("closing_balance"))
        result["total_income"] = to_float(result.get("total_income"))
        result["total_expenses"] = to_float(result.get("total_expenses"))
        result["overdraft_limit"] = to_float(result.get("overdraft_limit"))
        
        if "salary_credits" in result:
            result["salary_credits"] = [
                to_float(v) for v in result.get("salary_credits", [])
            ]
        
        bank_statement = BankStatementData(**result)
        required = [
        bank_statement.account_holder_name,
        bank_statement.bank_name,
        bank_statement.account_number,
        bank_statement.statement_months,
        bank_statement.statement_months,
        bank_statement.opening_balance,
        bank_statement.closing_balance,
        ]

        bank_statement.extraction_complete = all(required)

        return bank_statement
    
    
    def run(self, state):

        state.current_step = "Bank Statement Extraction"

        for document in state.documents:

            if document.document_type == "Bank Statement":
                bank_statement = self.extract(
                    document.extracted_text
                )
                state.bank_statements.append(bank_statement)

        state.completed_steps.append(
            "Bank Statement Extraction"
        )

        return state
