from agents.bank import BankVerificationAgent
from agents.employmentverification import EmploymentVerificationAgent
from agents.identity import IdentityVerificationAgent
from agents.income import IncomeVerificationAgent
from models.bank_statement import BankStatementData
from models.employment import EmploymentData
from models.passport import PassportData
from models.payslip import PayslipData
from workflow.state import MortgageState


class ForbiddenLLM:
    def chat_json(self, *_args, **_kwargs):
        raise AssertionError("unambiguous verification must not call the LLM")


def exact_state():
    return MortgageState(
        passport=PassportData(given_names="Jane", surname="Smith", extraction_complete=True),
        employment=EmploymentData(employer_name="Example Limited"),
        payslips=[
            PayslipData(employee_name="Jane Smith", employer_name="Example Ltd.", net_pay=2400),
            PayslipData(employee_name="JANE-SMITH", employer_name="Example Limited", net_pay=2400),
        ],
        bank_statements=[BankStatementData(
            account_holder_name="Jane Smith", account_number="12345678", sort_code="10-20-30",
            statement_months=3, salary_credits=[2400, 2400], overdraft_limit=0,
        )],
    )


def test_identity_exact_normalized_match_avoids_llm():
    state = exact_state()
    IdentityVerificationAgent(llm=ForbiddenLLM()).execute(state)
    assert state.identity.verified is True
    assert state.identity.confidence == 1


def test_income_consistent_figures_avoid_llm():
    state = exact_state()
    IncomeVerificationAgent(llm=ForbiddenLLM()).execute(state)
    assert state.income.verified is True
    assert state.income.monthly_income == 2400
    assert state.income.salary_credits == 2400


def test_bank_exact_identifiers_avoid_llm():
    state = exact_state()
    BankVerificationAgent(llm=ForbiddenLLM()).execute(state)
    assert state.bank.verified is True
    assert state.bank.statement_months == 3
    assert state.bank.overdraft is False


def test_ambiguous_employer_name_retains_semantic_fallback():
    state = exact_state()
    result = EmploymentVerificationAgent.deterministic_result(state)
    assert result is None
