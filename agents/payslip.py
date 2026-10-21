from agents.base_agent import BaseAgent
from models.payslip import PayslipData
from core.prompts import PAYSLIP_EXTRACTION_SYSTEM
from pydantic import BaseModel, ConfigDict


class PayslipExtraction(BaseModel):
    """Minimal model-facing schema; all keys required, unreadable values explicit."""

    model_config = ConfigDict(extra="forbid")
    employer_name: str | None
    employee_name: str | None
    gross_pay: float | None
    net_pay: float | None


class PayslipAgent(BaseAgent):
    name = "payslip_extraction"
    input_fields = ("documents",)
    output_fields = ("payslips",)

    def extract(self, text: str) -> list[PayslipData]:

        prompt = f"Payslip OCR text:\n{text}"

        result = self.chat_json(
            PAYSLIP_EXTRACTION_SYSTEM,
            prompt, list[PayslipExtraction]
        )
        
        payslips = []
        for item in result:
            payslips.append(PayslipData(**item, extraction_complete=all(
                item.get(key) is not None and item.get(key) != ""
                for key in ("employer_name", "employee_name", "gross_pay", "net_pay")
            )))

        return payslips
    
    
    def run(self, state):

        state.current_step = "Payslip Extraction"

        for document in state.documents:

            if document.document_type == "Payslip":
                payslip = self.extract(
                    document.extracted_text
                )
                state.payslips.extend(payslip)

        state.completed_steps.append(
            "Payslip Extraction"
        )

        return state
