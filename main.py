from pathlib import Path
import sys

from core.logging import configure_logging
from services.checks import render_check_table

from workflow.graph import MortgageWorkflow



def print_application_summary(state):
    print("\nAPPLICATION SUMMARY")
    print("=" * 60)

    print("Application ID:", state.application_id)
    print("Workflow Status:", state.workflow_status.value)

    print("Documents:", len(state.documents))

    if state.passport:

        print()

        print("Passport")

        print(state.passport)

    if state.employment:

        print()

        print("Employment")

        print(state.employment)
        
    if not state.payslips:
        print("No payslips extracted")
    else:
        for i, payslip in enumerate(state.payslips, start=1):
            print(f"\nPayslip {i}")
            #print(payslip)
            print(f"Employee: {payslip.employee_name}")
            print(f"Employer: {payslip.employer_name}")
            print(f"Pay Date: {payslip.pay_date}")
            print(f"Gross Pay: £{payslip.gross_pay}")
            print(f"Net Pay: £{payslip.net_pay}")
            print(f"Tax: £{payslip.tax}")
            print(f"NI: £{payslip.national_insurance}")
            print(f"Pension: £{payslip.pension}")
            print(f"Confidence: {payslip.confidence * 100:.0f}%")
            
    if not state.bank_statements:
        print("No Bank Statements extracted")
    else:
        for i, bank_statement in enumerate(state.bank_statements, start=1):
            print(f"\nBank Statement {i}")
            print(f"Account Holder: {bank_statement.account_holder_name}")
            print(f"Bank Name: {bank_statement.bank_name}")
            print(f"Account Number: {bank_statement.account_number}")
            print(f"Sort Code: {bank_statement.sort_code}")
            print(f"Statement Period: {bank_statement.statement_period}")
            print(f"Opening Balance: {bank_statement.opening_balance}")
            print(f"Closing Balance: {bank_statement.closing_balance}")
            print(f"Total Income: {bank_statement.total_income}")
            print(f"Total Expenses: {bank_statement.total_expenses}")
            print(f"Salary Credits: {bank_statement.salary_credits}")
            print(f"Overdraft Limit: {bank_statement.overdraft_limit}")
            print(f"Confidence: {bank_statement.confidence * 100:.0f}%")
            
        
    if state.document_ownership:

        print("\nDocument Ownership")

        print(state.document_ownership)
        
    if state.identity:

        print("\nIdentity")

        print(state.identity)
        
    if state.employmentverification:

        print("\nEmployment Verification")

        print(state.employmentverification)
        
    if state.income:

        print("\nIncome")

        print(state.income)
        
    if state.bank:

        print("\nBank Verification")

        print(state.bank)
        
    if state.fraud:

        print("\nFraud Assessment")

        print(state.fraud)
        
        
    if state.affordability:

        print("\nAffordability")

        print(state.affordability)
        
    if state.policy_engine:

        print("\nPolicy")

        print(state.policy_engine)
        
    if state.policy_assessment:

        print("\nPolicy Assessment Summary")

        print(state.policy_assessment)
        
    if state.underwriting:

        print("\nUnderwriting")

        print(state.underwriting)

    if state.human_review:
        print("\nHuman Review")
        print(state.human_review)

    if state.errors:
        print("\nWorkflow Errors")
        for error in state.errors:
            print(f"{error.stage} [{error.code}]: {error.message}")

    print("\nAPPLICATION CHECKS")
    print(render_check_table(state.check_results))
        
    print()

    print("Completed Steps")

    for step in state.completed_steps:

        print("✓", step)
        
        
        
        
def main():

    configure_logging()

    if len(sys.argv) != 2:
        print("Usage:")
        print("python main.py <application_folder>")
        sys.exit(1)

    application_folder = Path(sys.argv[1])

    if not application_folder.exists():
        print(f"Folder not found: {application_folder}")
        sys.exit(1)

    print("=" * 60)
    print("Mortgage AI Underwriting Copilot")
    print("=" * 60)
    print()

    workflow = MortgageWorkflow()
    
    state = workflow.run(application_folder )

    
    print_application_summary(state)
        

if __name__ == "__main__":
    main()
