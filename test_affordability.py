import statistics
from datetime import datetime

bank_statements=[{
  "bank_name": "HSBC",
  "account_holder_name": "John Robert Smith",
  "account_number": "",
  "sort_code": "49",
  "statement_period": "10 April to 9 July 2015",
  "opening_balance": -100.70,
  "closing_balance": -111.78,
  "total_income": 4233.73,
  "total_expenses": 4244.81,
  "salary_credits": [1999.36],
  "overdraft_limit": 750.00,
  "confidence": 0.8
}]

payslips=[{
    "employer_name": "ABC Corporation Ltd",
    "employee_name": "John Robert Smith",
    "pay_date": "6/09/22",
    "pay_period": "Month 6",
    "gross_pay": 4118.00,
    "net_pay": 2953.71,
    "tax": 613.60,
    "national_insurance": 406.77,
    "pension": 143.92,
    "tax_code": "1259L",
    "ni_number": "AB123456C",
    "confidence": 0.98
  },
    {
    "employer_name": "ABC Corporation Ltd",
    "employee_name": "Timothy Jones",
    "pay_date": "30/09/22",
    "pay_period": "Month 6",
    "gross_pay": 3250.00,
    "net_pay": 2510.97,
    "tax": 338.07,
    "national_insurance": 291.76,
    "pension": 109.20,
    "tax_code": "1000L",
    "ni_number": "AB987654D",
    "confidence": 0.98
  },
  {
    "employer_name": "ABC Corporation Ltd",
    "employee_name": "Mary Anne Wright",
    "pay_date": "6/09/22",
    "pay_period": "Week 24",
    "gross_pay": 1837.50,
    "net_pay": 1667.86,
    "tax": 0.00,
    "national_insurance": 115.34,
    "pension": 54.30,
    "tax_code": "1257L",
    "ni_number": "AB564122C",
    "confidence": 0.98
  }
]

print(bank_statements[0]["statement_period"])
start_date = bank_statements[0]["statement_period"].partition("to")
end_date = start_date[-1].strip()
start_date = start_date[0].strip()
start_date_object = datetime.strptime(start_date, "%d %B").date()
end_date_object = datetime.strptime(end_date, "%d %B %Y").date()
dummy_year = 2000 
start_date_same_year = start_date_object.replace(year=dummy_year)
end_date_same_year = end_date_object.replace(year=dummy_year)
# Subtract to get a timedelta object and extract total months
days_difference = abs((end_date_same_year - start_date_same_year).days)
months_difference = days_difference/30
print(start_date)
print(end_date)
print(start_date_object)
print(end_date_object)
print(months_difference)
months = months_difference

avg_income = statistics.mean([
    p["net_pay"] for p in payslips
])
#
avg_expenses = (
    sum(bs["total_expenses"] for bs in bank_statements)
    / months
)
#
disposable = avg_income - avg_expenses
#
dti = avg_expenses / avg_income

print(avg_income)
print(avg_expenses)
print(disposable)
print(dti)