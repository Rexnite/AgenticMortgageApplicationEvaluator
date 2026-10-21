from pydantic import BaseModel

class EmploymentData(BaseModel):
    employer_name: str | None = None
    job_title: str | None = None
    employment_type: str | None = None

    annual_salary: float | None = None

    weekly_hours: float | None = None

    start_date: str | None = None

    probation: str | None = None

    confidence: float = 0.0
        
    extraction_complete: bool = False