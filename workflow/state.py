from pydantic import BaseModel, Field
from uuid import uuid4
from typing import Any

from models.document import Document
from models.check_result import CheckResult
from models.passport import PassportData
from models.payslip import PayslipData
from models.employment import EmploymentData
from models.document_ownership import DocumentOwnership
from models.bank_statement import BankStatementData
from models.affordability import AffordabilityData
from models.underwriting import UnderwritingDecision
from models.identity import IdentityVerification
from models.employmentverification import EmploymentVerification
from models.income import IncomeVerification
from models.bank import BankVerification
from models.fraud import FraudAssessment
from models.policy_engine import PolicyAssessment, PolicyResult, PolicyRule
from models.policy_assessment import PolicyAssessmentSummary
from models.policy_retrieval import PolicyDocument
from models.workflow import HumanReview, HumanReviewDecision, ReflectionResult, WorkflowError, WorkflowStatus


class MortgageState(BaseModel):

    application_id: str = Field(default_factory=lambda: str(uuid4()))

    documents: list[Document] = Field(default_factory=list)

    check_results: list[CheckResult] = Field(default_factory=list)

    passport: PassportData | None = None

    payslips: list[PayslipData] = Field(default_factory=list)

    bank_statements: list[BankStatementData] = Field(default_factory=list)

    employment: EmploymentData | None = None
        
    document_ownership: list[DocumentOwnership] = Field(default_factory=list)
        
    rejected_documents: list[Any] = Field(default_factory=list)

    affordability: AffordabilityData | None = None

    underwriting: UnderwritingDecision | None = None
        
    identity: IdentityVerification | None = None
        
    employmentverification: EmploymentVerification | None = None
        
    income: IncomeVerification | None = None
        
    bank: BankVerification | None = None
        
    fraud: FraudAssessment | None = None
        
    policy_rules: list[PolicyRule] = Field(default_factory=list)
        
    policy_results: list[PolicyResult] = Field(default_factory=list)
        
    policy_engine: PolicyAssessment | None = None
        
    policy_assessment: PolicyAssessmentSummary | None = None

    policy_context: list[PolicyDocument] = Field(default_factory=list)

    reflection: ReflectionResult | None = None

    human_review: HumanReview | None = None

    review_decision: HumanReviewDecision | None = None

    workflow_status: WorkflowStatus = WorkflowStatus.RUNNING

    route: str = "START"

    current_step: str = "START"

    completed_steps: list[str] = Field(default_factory=list)

    errors: list[WorkflowError] = Field(default_factory=list)
