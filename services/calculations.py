from datetime import date, datetime
from statistics import mean
from typing import Iterable


def safe_mean(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return mean(usable) if usable else None


def monthly_to_annual(monthly_income: float | None) -> float | None:
    if monthly_income is None or monthly_income < 0:
        return None
    return float(monthly_income) * 12


def annual_to_monthly(annual_income: float | None) -> float | None:
    if annual_income is None or annual_income < 0:
        return None
    return float(annual_income) / 12


def calculate_age(date_of_birth: str | date | None, as_of: date | None = None) -> int | None:
    if date_of_birth is None:
        return None
    if isinstance(date_of_birth, str):
        parsed = None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                parsed = datetime.strptime(date_of_birth, fmt).date()
                break
            except ValueError:
                continue
        if parsed is None:
            return None
        date_of_birth = parsed
    today = as_of or date.today()
    return today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))


def average_monthly_income(payslips, gross: bool = False) -> float | None:
    field = "gross_pay" if gross else "net_pay"
    return safe_mean(getattr(item, field, None) for item in payslips)


def annual_income(employment, payslips) -> float | None:
    stated = getattr(employment, "annual_salary", None) if employment else None
    if stated is not None:
        return float(stated)
    monthly = average_monthly_income(payslips, gross=True)
    return monthly_to_annual(monthly)


def average_payslip_income(payslips, gross: bool = False) -> float | None:
    """Explicit domain alias used by verification and affordability agents."""
    return average_monthly_income(payslips, gross=gross)


def average_monthly_expenses(bank_statements) -> float | None:
    monthly_values = []
    for statement in bank_statements:
        expenses = getattr(statement, "total_expenses", None)
        months = getattr(statement, "statement_months", None)
        if expenses is not None and months and months > 0:
            monthly_values.append(float(expenses) / float(months))
    return safe_mean(monthly_values)


def recurring_salary_credit(bank_statements) -> float | None:
    return safe_mean(
        credit
        for statement in bank_statements
        for credit in (getattr(statement, "salary_credits", None) or [])
    )


def debt_to_income(monthly_debt: float | None, monthly_gross_income: float | None) -> float | None:
    if monthly_debt is None or not monthly_gross_income or monthly_gross_income <= 0:
        return None
    return float(monthly_debt) / float(monthly_gross_income)


def compare_income_sources(first: float | None, second: float | None, tolerance: float = 0.10) -> bool | None:
    """Compare two positive income figures within a relative tolerance."""
    if first is None or second is None or first <= 0 or second <= 0 or tolerance < 0:
        return None
    return abs(float(first) - float(second)) / max(abs(float(first)), abs(float(second))) <= tolerance
