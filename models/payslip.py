from pydantic import BaseModel

class PayslipData(BaseModel):
    employer_name: str | None = None
    employee_name: str | None = None

    pay_date: str | None = None
    pay_period: str | None = None

    gross_pay: float | None = None
    net_pay: float | None = None

    tax: float | None = None
    national_insurance: float | None = None
    pension: float | None = None

    tax_code: str | None = None
    ni_number: str | None = None

    confidence: float = 0.0
        
    extraction_complete: bool = False
