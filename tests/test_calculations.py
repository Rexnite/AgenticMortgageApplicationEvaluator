from datetime import date

from models.bank_statement import BankStatementData
from models.employment import EmploymentData
from models.payslip import PayslipData
from services.calculations import (
    annual_income, annual_to_monthly, average_monthly_expenses,
    average_monthly_income, average_payslip_income, calculate_age,
    compare_income_sources, debt_to_income, monthly_to_annual,
)


def test_age_handles_birthday_boundary():
    assert calculate_age("2000-09-01", date(2026, 8, 29)) == 25
    assert calculate_age("2000-08-01", date(2026, 8, 29)) == 26
    assert calculate_age("not-a-date") is None


def test_income_averages_and_annual_fallback():
    payslips = [PayslipData(gross_pay=3000, net_pay=2400), PayslipData(gross_pay=3200, net_pay=2500)]
    assert average_monthly_income(payslips) == 2450
    assert average_monthly_income(payslips, gross=True) == 3100
    assert annual_income(None, payslips) == 37200
    assert annual_income(EmploymentData(annual_salary=40000), payslips) == 40000


def test_expenses_and_dti_are_deterministic():
    statements = [BankStatementData(total_expenses=3000, statement_months=3), BankStatementData(total_expenses=2200, statement_months=2)]
    assert average_monthly_expenses(statements) == 1050
    assert debt_to_income(1050, 3000) == 0.35
    assert debt_to_income(None, 3000) is None
    assert debt_to_income(100, 0) is None


def test_missing_values_return_none():
    assert average_monthly_income([]) is None
    assert average_monthly_expenses([BankStatementData(total_expenses=10, statement_months=None)]) is None


def test_income_conversions_and_boundaries():
    assert annual_to_monthly(36000) == 3000
    assert monthly_to_annual(3000) == 36000
    assert annual_to_monthly(0) == 0
    assert monthly_to_annual(0) == 0
    assert annual_to_monthly(None) is None
    assert monthly_to_annual(-1) is None


def test_average_payslip_income_ignores_missing_values():
    payslips = [PayslipData(net_pay=2000), PayslipData(net_pay=None), PayslipData(net_pay=2200)]
    assert average_payslip_income(payslips) == 2100


def test_zero_debt_and_income_source_tolerance():
    assert debt_to_income(0, 3000) == 0
    assert compare_income_sources(3000, 2700, tolerance=.10) is True
    assert compare_income_sources(3000, 2600, tolerance=.10) is False
    assert compare_income_sources(0, 100) is None
    assert compare_income_sources(None, 100) is None
