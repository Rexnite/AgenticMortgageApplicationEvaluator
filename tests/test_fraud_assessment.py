from pathlib import Path

from agents.fraud import FraudAssessmentAgent
from agents.policy_assessment import PolicyAssessmentSummaryAgent
from agents.policy_engine import PolicyAssessmentAgent
from agents.reflection import ReflectionAgent
from agents.underwriter import UnderwriterAgent
from models.affordability import AffordabilityData
from models.bank import BankVerification
from models.bank_statement import BankStatementData
from models.document import Document
from models.employment import EmploymentData
from models.employmentverification import EmploymentVerification
from models.fraud import FraudAssessment, FraudSignalStatus
from models.identity import IdentityVerification
from models.income import IncomeVerification
from models.passport import PassportData
from models.payslip import PayslipData
from models.workflow import ReflectionResult
from workflow.state import MortgageState


def consistent_state():
    return MortgageState(
        passport=PassportData(given_names="Jane", surname="Smith", age=35, extraction_complete=True),
        employment=EmploymentData(employer_name="Example Ltd", annual_salary=48000, extraction_complete=True),
        payslips=[
            PayslipData(employee_name="Jane Smith", employer_name="Example Ltd", gross_pay=4000, net_pay=3000, extraction_complete=True),
            PayslipData(employee_name="JANE-SMITH", employer_name="Example Ltd", gross_pay=4000, net_pay=3000, extraction_complete=True),
        ],
        bank_statements=[
            BankStatementData(account_holder_name="Jane Smith", account_number="12345678", sort_code="10-20-30", statement_months=1, salary_credits=[3000], extraction_complete=True),
            BankStatementData(account_holder_name="JANE SMITH", account_number="12345678", sort_code="102030", statement_months=1, salary_credits=[3000], extraction_complete=True),
        ],
        documents=[Document(file_name="evidence.pdf", file_path=Path("evidence.pdf"), extension=".pdf", size=10,
            document_type="Payslip", extracted_text="readable evidence", extraction_method="pdf_text",
            extraction_success=True, extraction_confidence=1, page_count=1)],
        identity=IdentityVerification(applicant_name="Jane Smith", verified=True, inconsistencies=[], missing_information=[], confidence=1),
        employmentverification=EmploymentVerification(applicant_name="Jane Smith", verified=True, employer="Example Ltd", inconsistencies=[], missing_information=[], confidence=1),
        income=IncomeVerification(applicant_name="Jane Smith", verified=True, monthly_income=3000, salary_credits=3000, inconsistencies=[], missing_information=[], confidence=1),
        bank=BankVerification(applicant_name="Jane Smith", verified=True, statement_months=2, overdraft=False, inconsistencies=[], missing_information=[], confidence=1),
        affordability=AffordabilityData(average_monthly_net_income=3000, debt_to_income_ratio=.3, affordability_score=80, confidence=1),
    )


def signal(state, code):
    assessment = FraudAssessmentAgent(llm=object()).assess(state, use_llm=False)
    return assessment, next(item for item in assessment.signals if item.code == code)


def test_clean_evidence_has_no_artificial_active_fraud_signals():
    assessment = FraudAssessmentAgent(llm=object()).assess(consistent_state(), use_llm=False)
    assert assessment.fraud_score == 0
    assert assessment.risk == "Low"
    assert not [item for item in assessment.signals
        if item.status in {FraudSignalStatus.DETECTED, FraudSignalStatus.SUSPECTED}]


def test_identity_inconsistency_creates_observed_signal_without_claiming_forgery():
    state = consistent_state()
    state.identity.verified = False
    state.identity.inconsistencies = ["Names conflict"]
    assessment, item = signal(state, "IDENTITY_VERIFICATION_CONFLICT")
    assert item.status == FraudSignalStatus.DETECTED
    assert item.description in assessment.confirmed_inconsistencies
    assert "forgery" not in item.description.casefold()


def test_income_mismatch_is_detected_deterministically():
    state = consistent_state()
    for statement in state.bank_statements:
        statement.salary_credits = [1000]
    _, item = signal(state, "PAYSLIP_BANK_INCOME_CONSISTENCY")
    assert item.status == FraudSignalStatus.DETECTED


def test_account_inconsistency_is_detected_when_comparable():
    state = consistent_state()
    state.bank_statements[1].account_number = "87654321"
    _, item = signal(state, "BANK_ACCOUNT_NUMBER_CONSISTENCY")
    assert item.status == FraudSignalStatus.DETECTED


def test_account_holder_inconsistency_is_detected_when_comparable():
    state = consistent_state()
    state.bank_statements[1].account_holder_name = "Different Person"
    _, item = signal(state, "BANK_ACCOUNT_HOLDER_CONSISTENCY")
    assert item.status == FraudSignalStatus.DETECTED


def test_missing_evidence_is_not_assessed_not_passed():
    state = MortgageState(passport=PassportData(), employment=EmploymentData())
    assessment = FraudAssessmentAgent(llm=object()).assess(state, use_llm=False)
    assert assessment.fraud_score == 0
    assert assessment.unassessed_checks
    assert next(item for item in assessment.signals
        if item.code == "PAYSLIP_BANK_INCOME_CONSISTENCY").status == FraudSignalStatus.NOT_ASSESSED


def test_unreadable_document_is_quality_issue_not_tampering_claim():
    state = consistent_state()
    state.documents[0].extraction_success = False
    state.documents[0].extracted_text = ""
    assessment, quality = signal(state, "DOCUMENT_EXTRACTION_QUALITY")
    tampering = next(item for item in assessment.signals if item.code == "DOCUMENT_TAMPERING_FORENSICS")
    assert quality.status == FraudSignalStatus.DETECTED
    assert tampering.status == FraudSignalStatus.NOT_ASSESSED


def test_fraud_policy_threshold_remains_authoritative():
    state = consistent_state()
    state.fraud = FraudAssessment(applicant_name="Jane Smith", fraud_score=61, risk="Medium", confidence=1)
    PolicyAssessmentAgent().run(state)
    result = next(item for item in state.policy_results if item.rule_id == "FRAUD001")
    assert result.status == "REFER"


def test_reflection_flags_unsupported_high_fraud_conclusion():
    state = consistent_state()
    state.fraud = FraudAssessment(applicant_name="Jane Smith", fraud_score=80, risk="High",
        indicators=["Unsupported accusation"], confidence=1)
    PolicyAssessmentAgent().run(state)
    PolicyAssessmentSummaryAgent().run(state)
    result = ReflectionAgent().assess(state)
    assert result.passed is False
    assert "Fraud conclusion is not supported by structured active signals" in result.policy_inconsistencies


class CapturingLLM:
    def __init__(self):
        self.prompt = ""

    def chat_json(self, _system, prompt):
        self.prompt = prompt
        return {"decision": "Approve", "risk_rating": "Low", "confidence": 1,
            "strengths": [], "concerns": [], "conditions": [], "summary": "Reviewed"}


def test_underwriter_receives_structured_fraud_semantics_and_guardrail():
    state = consistent_state()
    state.fraud = FraudAssessmentAgent(llm=object()).assess(state, use_llm=False)
    PolicyAssessmentAgent().run(state)
    PolicyAssessmentSummaryAgent().run(state)
    state.reflection = ReflectionResult()
    llm = CapturingLLM()
    UnderwriterAgent(llm=llm).run(state)
    assert '"signals"' in llm.prompt
    assert "NOT_ASSESSED means evidence or technical capability was unavailable" in llm.prompt
    assert "Do not claim document tampering" in llm.prompt


class SynthesisLLM:
    def chat_json(self, _system, prompt):
        assert "Active structured signals" in prompt
        assert "Do not create new signals" in prompt
        return {"suspected_concerns": ["Contextual review recommended"],
            "recommended_action": "Manual review", "confidence": .7,
            "fraud_score": 100}


def test_semantic_synthesis_cannot_override_deterministic_score():
    state = consistent_state()
    state.income.verified = False
    state.income.inconsistencies = ["Mismatch"]
    agent = FraudAssessmentAgent(llm=SynthesisLLM())
    deterministic = agent.assess(state, use_llm=False)
    semantic = agent.assess(state, use_llm=True)
    assert semantic.fraud_score == deterministic.fraud_score
    assert semantic.recommended_action == "Manual review"
