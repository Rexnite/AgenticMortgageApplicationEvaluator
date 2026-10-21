from pathlib import Path

import pytest

from agents.base_agent import BaseAgent
from agents.policy_assessment import PolicyAssessmentSummaryAgent
from agents.policy_engine import PolicyAssessmentAgent
from models.affordability import AffordabilityData
from models.bank import BankVerification
from models.bank_statement import BankStatementData
from models.document import Document
from models.employment import EmploymentData
from models.employmentverification import EmploymentVerification
from models.fraud import FraudAssessment
from models.identity import IdentityVerification
from models.income import IncomeVerification
from models.passport import PassportData
from models.payslip import PayslipData
from models.underwriting import UnderwritingDecision
from models.workflow import ErrorCategory, ReflectionResult
from workflow.graph import MortgageWorkflow


class FunctionAgent(BaseAgent):
    name = "test_agent"
    def __init__(self, function=lambda state: state):
        super().__init__(llm=object())
        self.function = function
    def run(self, state, *args):
        return self.function(state)


def docs(complete=True):
    types = ["Passport", "Employment Letter", "Payslip", "Bank Statement"] if complete else ["Passport"]
    return [Document(file_name=f"{kind}.pdf", file_path=Path(f"{kind}.pdf"), extension=".pdf", size=1, document_type=kind) for kind in types]


def fake_agents(*, complete=True, age=30, identity=True, fraud=10, dti=.3, score=80,
        fail_stage=None, missing_extraction=None, reflection_passed=True, calls=None):
    calls = calls if calls is not None else []
    def intake(state): state.documents = docs(complete); return state
    def passport(state):
        state.passport = PassportData(age=age, extraction_complete=True)
        state.completed_steps.append("Passport Extraction")
        return state
    def employment(state):
        state.employment = EmploymentData(annual_salary=48000, extraction_complete=True)
        state.completed_steps.append("Employment Extraction")
        return state
    def payslip(state):
        state.payslips = [] if missing_extraction == "payslip" else [PayslipData(employee_name="A", gross_pay=4000, net_pay=3000, extraction_complete=True)]
        state.completed_steps.append("Payslip Extraction")
        return state
    def bank_statement(state):
        state.bank_statements = [] if missing_extraction == "bank_statement" else [BankStatementData(account_holder_name="A", statement_months=3, total_expenses=1000, salary_credits=[3000], extraction_complete=True)]
        state.completed_steps.append("Bank Statement Extraction")
        return state
    def verify_identity(state): state.identity = IdentityVerification(applicant_name="A", verified=identity, inconsistencies=[], missing_information=[], confidence=1); return state
    def verify_employment(state): state.employmentverification = EmploymentVerification(applicant_name="A", verified=True, employer="E", inconsistencies=[], missing_information=[], confidence=1); return state
    def income(state): state.income = IncomeVerification(applicant_name="A", verified=True, monthly_income=3000, salary_credits=3000, inconsistencies=[], missing_information=[], confidence=1); return state
    def bank(state): state.bank = BankVerification(applicant_name="A", verified=True, statement_months=3, overdraft=False, inconsistencies=[], missing_information=[], confidence=1); return state
    def fraud_agent(state): state.fraud = FraudAssessment(applicant_name="A", fraud_score=fraud, risk="Low", indicators=[], confidence=1); return state
    def affordability(state): state.affordability = AffordabilityData(average_monthly_net_income=3000, debt_to_income_ratio=dti, affordability_score=score, confidence=1); return state
    def reflection(state): state.reflection = ReflectionResult(passed=reflection_passed, issues=[] if reflection_passed else ["injected contradiction"]); return state
    def underwriter(state): calls.append("underwriter"); state.underwriting = UnderwritingDecision(decision="Approve", confidence=1, risk_rating="Low"); state.route = "APPROVE"; return state
    functions = {"document": intake, "passport": passport, "employment": employment, "payslip": payslip,
        "bank_statement": bank_statement, "ownership": lambda s:s, "identity": verify_identity,
        "employment_verification": verify_employment, "income": income, "bank": bank,
        "fraud": fraud_agent, "affordability": affordability, "reflection": reflection, "underwriter": underwriter}
    if fail_stage:
        def fail(_state): raise RuntimeError("injected failure")
        functions[fail_stage] = fail
    return {name: FunctionAgent(fn) for name, fn in functions.items()}


def run(**kwargs):
    return MortgageWorkflow(agents=fake_agents(**kwargs)).run("unused")


def test_happy_path_reaches_final_underwriter():
    state = run()
    assert state.underwriting.decision == "Approve"
    assert state.reflection.passed
    assert state.human_review is None


def test_workflow_injects_one_session_scoped_llm_into_default_agents():
    sentinel = object()
    workflow = MortgageWorkflow(llm=sentinel)
    assert workflow.agents
    assert all(agent._llm is sentinel for agent in workflow.agents.values())


@pytest.mark.parametrize("kwargs,expected,status", [
    ({"complete": False}, "Required documents are missing", "REQUEST_MORE_INFORMATION"),
    ({"fraud": 80}, "Policy rules require referral", "PENDING_REVIEW"),
    ({"dti": .5}, "Policy rules require referral", "PENDING_REVIEW"),
])
def test_referral_routes(kwargs, expected, status):
    state = run(**kwargs)
    assert state.human_review.reason == expected
    assert state.human_review.status == status


def test_mandatory_policy_failure_declines_without_llm():
    state = run(age=17)
    assert state.route == "DECLINE"
    assert state.underwriting.decision == "Decline"


def test_agent_exception_routes_to_structured_error_review():
    state = run(fail_stage="passport")
    assert state.errors[0].stage == "Extraction"
    assert state.errors[0].category == ErrorCategory.EXTRACTION_FAILURE
    assert state.errors[0].recoverable is True
    assert state.human_review.reason == "Workflow execution failed"


def test_graph_exposes_distinct_langgraph_nodes():
    workflow = MortgageWorkflow(agents=fake_agents())
    graph = workflow.build_graph(Path("unused")).get_graph()
    assert {
        "intake", "extraction", "ownership", "identity_verification",
        "employment_verification", "income_verification", "bank_verification",
        "fraud", "affordability", "policy_evaluation", "policy_summary",
        "reflection", "underwriter", "human_review", "decline", "error",
    }.issubset(graph.nodes)


def test_complete_documents_with_missing_extraction_request_information():
    state = run(missing_extraction="bank_statement")
    assert state.route == "HUMAN_REVIEW"
    assert state.human_review.status == "REQUEST_MORE_INFORMATION"
    assert state.human_review.reason == "Required evidence could not be extracted"


@pytest.mark.parametrize("kwargs", [{"fraud": 80}, {"age": 17}])
def test_policy_terminal_paths_do_not_call_underwriter(kwargs):
    calls = []
    state = MortgageWorkflow(agents=fake_agents(calls=calls, **kwargs)).run("unused")
    assert calls == []
    assert state.route in {"HUMAN_REVIEW", "DECLINE"}


def test_reflection_escalation_does_not_call_underwriter():
    calls = []
    state = MortgageWorkflow(agents=fake_agents(calls=calls, reflection_passed=False)).run("unused")
    assert calls == []
    assert state.route == "HUMAN_REVIEW"
    assert state.human_review.reason == "Quality-control checks found inconsistencies"


def test_unrecoverable_infrastructure_failure_uses_error_terminal():
    calls = []
    state = MortgageWorkflow(agents=fake_agents(calls=calls, fail_stage="affordability")).run("unused")
    assert calls == []
    assert state.route == "ERROR"
    assert state.current_step == "Error"
    assert state.human_review is None
    assert state.errors[0].category == ErrorCategory.INFRASTRUCTURE_FAILURE
    assert state.errors[0].recoverable is False


@pytest.mark.parametrize(
    "kwargs,expected_status,expected_route",
    [
        ({}, "PASS", "reflect"),
        ({"dti": .5}, "REFER", "review"),
        ({"age": 17}, "FAIL", "decline"),
    ],
)
def test_policy_routes_are_logged(caplog, kwargs, expected_status, expected_route):
    state = MortgageWorkflow._state({})
    values = {"age": 30, "dti": .3, **kwargs}
    state.passport = PassportData(age=values["age"])
    state.identity = IdentityVerification(applicant_name="A", verified=True, inconsistencies=[], missing_information=[], confidence=1)
    state.fraud = FraudAssessment(applicant_name="A", fraud_score=10, risk="Low", indicators=[], confidence=1)
    state.affordability = AffordabilityData(debt_to_income_ratio=values["dti"], affordability_score=80)
    PolicyAssessmentAgent().run(state)
    PolicyAssessmentSummaryAgent().run(state)

    workflow = MortgageWorkflow.__new__(MortgageWorkflow)
    with caplog.at_level("INFO", logger="mortgage.workflow"):
        selected = workflow._policy_route(state)

    assert selected == expected_route
    route_record = next(record for record in caplog.records if record.msg == "route_selected")
    assert route_record.route_decision == expected_route
    assert route_record.route_reason == "policy_outcome"
    assert route_record.policy_status == expected_status
