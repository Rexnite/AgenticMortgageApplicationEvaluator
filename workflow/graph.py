import logging
from pathlib import Path
from time import perf_counter

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from agents.affordability import AffordabilityAgent
from agents.base_agent import AgentContractError
from agents.bank import BankVerificationAgent
from agents.bank_statement import BankStatementAgent
from agents.document import DocumentAgent
from agents.document_ownership import DocumentOwnershipAgent
from agents.employment import EmploymentAgent
from agents.employmentverification import EmploymentVerificationAgent
from agents.fraud import FraudAssessmentAgent
from agents.identity import IdentityVerificationAgent
from agents.income import IncomeVerificationAgent
from agents.passport import PassportAgent
from agents.payslip import PayslipAgent
from agents.policy_assessment import PolicyAssessmentSummaryAgent
from agents.policy_engine import PolicyAssessmentAgent
from agents.reflection import ReflectionAgent
from agents.underwriter import UnderwriterAgent
from core.logging import state_summary
from core.llm import LocalModelUnavailable, StructuredOutputError
from services.checks import build_check_results
from models.underwriting import UnderwritingDecision
from models.workflow import (
    ErrorCategory, HumanReview, HumanReviewDecision, ReviewDecision, ReviewStatus,
    WorkflowError, WorkflowStatus,
)
from workflow.state import MortgageState


LOGGER = logging.getLogger("mortgage.workflow")
REQUIRED_DOCUMENTS = {"Passport", "Employment Letter", "Payslip", "Bank Statement"}
EXTRACTION_AGENTS = {"document", "passport", "employment", "payslip", "bank_statement"}
SEMANTIC_AGENTS = {"ownership", "identity", "employment_verification", "income", "bank", "fraud", "underwriter"}
POLICY_AGENTS = {"policy", "policy_summary"}


class MortgageWorkflow:
    """Conditional LangGraph orchestration around the existing specialist agents."""

    def __init__(self, agents=None, required_documents=None, checkpointer=None, llm=None):
        defaults = {
            "document": DocumentAgent(llm=llm), "passport": PassportAgent(llm=llm),
            "employment": EmploymentAgent(llm=llm), "payslip": PayslipAgent(llm=llm),
            "bank_statement": BankStatementAgent(llm=llm), "ownership": DocumentOwnershipAgent(llm=llm),
            "identity": IdentityVerificationAgent(llm=llm), "employment_verification": EmploymentVerificationAgent(llm=llm),
            "income": IncomeVerificationAgent(llm=llm), "bank": BankVerificationAgent(llm=llm),
            "fraud": FraudAssessmentAgent(llm=llm), "affordability": AffordabilityAgent(llm=llm),
            "policy": PolicyAssessmentAgent(llm=llm), "policy_summary": PolicyAssessmentSummaryAgent(llm=llm),
            "reflection": ReflectionAgent(llm=llm), "underwriter": UnderwriterAgent(llm=llm),
        }
        defaults.update(agents or {})
        self.agents = defaults
        self.required_documents = frozenset(required_documents or REQUIRED_DOCUMENTS)
        self.checkpointer = checkpointer
        self.graph = None

    @staticmethod
    def _state(value):
        return value if isinstance(value, MortgageState) else MortgageState.model_validate(value)

    def _node(self, name, agent_names, *args):
        def invoke(value):
            state = self._state(value)
            started = perf_counter()
            LOGGER.info("stage_started", extra={"stage": name, "application_id": state.application_id, **state_summary(state)})
            try:
                for index, agent_name in enumerate(agent_names):
                    call_args = args if index == 0 else ()
                    state = self.agents[agent_name].execute(state, *call_args)
                LOGGER.info("stage_completed", extra={"stage": name, "application_id": state.application_id,
                    "duration_ms": round((perf_counter() - started) * 1000, 2), **state_summary(state)})
            except Exception as exc:
                category, recoverable = self._classify_error(exc, agent_name)
                code = "LOCAL_MODEL_UNAVAILABLE" if isinstance(exc, LocalModelUnavailable) else type(exc).__name__
                state.errors.append(WorkflowError(stage=name, agent=agent_name, code=code, message=str(exc), recoverable=recoverable, category=category))
                state.route = "ERROR"
                LOGGER.exception("stage_failed", extra={"stage": name, "agent": agent_name, "application_id": state.application_id,
                    "error_code": type(exc).__name__, "error_category": category.value,
                    "recoverable": recoverable, "duration_ms": round((perf_counter() - started) * 1000, 2)})
            state.check_results = build_check_results(state, getattr(self, "required_documents", REQUIRED_DOCUMENTS))
            return state.model_dump(mode="python")
        return invoke

    @staticmethod
    def _classify_error(exc, agent_name):
        if isinstance(exc, LocalModelUnavailable):
            return ErrorCategory.LLM_FAILURE, True
        if isinstance(exc, StructuredOutputError):
            return ErrorCategory.SCHEMA_VALIDATION_FAILURE, True
        if isinstance(exc, AgentContractError):
            return ErrorCategory.MISSING_INPUT, True
        if isinstance(exc, ValidationError):
            return ErrorCategory.SCHEMA_VALIDATION_FAILURE, True
        if agent_name in EXTRACTION_AGENTS:
            return ErrorCategory.EXTRACTION_FAILURE, True
        if agent_name in SEMANTIC_AGENTS:
            return ErrorCategory.LLM_FAILURE, True
        if agent_name in POLICY_AGENTS:
            return ErrorCategory.POLICY_FAILURE, False
        return ErrorCategory.INFRASTRUCTURE_FAILURE, False

    def _missing_documents(self, state):
        return sorted(self.required_documents - {document.document_type for document in state.documents})

    @staticmethod
    def _missing_extracted_evidence(state):
        missing = []
        if state.passport is None:
            missing.append("Passport extraction")
        if state.employment is None:
            missing.append("Employment letter extraction")
        if not state.payslips:
            missing.append("Payslip extraction")
        if not state.bank_statements:
            missing.append("Bank statement extraction")
        return missing

    def _route_errors(self, state, source):
        decision = "review" if state.errors and all(error.recoverable for error in state.errors) else "error"
        reason = "recoverable_workflow_error" if decision == "review" else "unrecoverable_workflow_error"
        self._log_route(state, source, decision, reason,
            error_categories=[error.category.value for error in state.errors])
        return decision

    def _intake_route(self, value):
        state = self._state(value)
        if state.errors:
            return self._route_errors(state, "Document Intake")
        missing = self._missing_documents(state)
        if missing:
            self._log_route(state, "Document Intake", "review", "missing_required_documents", missing_document_types=missing)
            return "review"
        self._log_route(state, "Document Intake", "extract", "required_documents_present")
        return "extract"

    def _extraction_route(self, value):
        state = self._state(value)
        if state.errors:
            return self._route_errors(state, "Extraction")
        missing = self._missing_extracted_evidence(state)
        if missing:
            self._log_route(state, "Extraction", "review", "missing_extracted_evidence", missing_evidence=missing)
            return "review"
        self._log_route(state, "Extraction", "continue", "critical_evidence_extracted")
        return "continue"

    def _standard_route(self, value, source=None):
        state = self._state(value)
        if state.errors:
            return self._route_errors(state, source or state.current_step)
        self._log_route(state, source or state.current_step, "continue", "stage_success")
        return "continue"

    def _ownership_route(self, value):
        state = self._state(value)
        if state.errors:
            return self._route_errors(state, "Document Ownership")
        decision = "review" if state.rejected_documents else "continue"
        self._log_route(state, "Document Ownership", decision, "rejected_documents" if state.rejected_documents else "ownership_confirmed")
        return decision

    def _policy_route(self, value):
        state = self._state(value)
        if state.errors:
            return self._route_errors(state, "Policy Summary")
        status = state.policy_assessment.overall_status if state.policy_assessment else None
        decision = {"FAIL": "decline", "REFER": "review", "PASS": "reflect"}.get(status, "error")
        self._log_route(state, "Policy", decision, "policy_outcome", policy_status=status)
        return decision

    def _reflection_route(self, value):
        state = self._state(value)
        if state.errors:
            return self._route_errors(state, "Quality Control")
        decision = "continue" if state.reflection and state.reflection.passed else "review"
        self._log_route(state, "Quality Control", decision, "quality_control_passed" if decision == "continue" else "quality_control_concerns")
        return decision

    @staticmethod
    def _log_route(state, source, decision, reason, **details):
        LOGGER.info("route_selected", extra={"application_id": state.application_id, "source": source,
            "route_decision": decision, "route_reason": reason, **details, **state_summary(state)})

    def _human_review(self, value):
        state = self._state(value)
        missing = self._missing_documents(state)
        missing_evidence = self._missing_extracted_evidence(state) if not missing else []
        status = ReviewStatus.PENDING_REVIEW
        if missing:
            reason, uncertain, action = "Required documents are missing", missing, "Request the missing documents"
            status = ReviewStatus.REQUEST_MORE_INFORMATION
        elif state.errors:
            reason = "Workflow execution failed"
            uncertain = [f"{error.stage}: {error.message}" for error in state.errors]
            action = "Resolve the error and rerun the affected stage"
            if any(error.category in {ErrorCategory.MISSING_INPUT, ErrorCategory.EXTRACTION_FAILURE} for error in state.errors):
                status = ReviewStatus.REQUEST_MORE_INFORMATION
        elif state.rejected_documents:
            reason, uncertain, action = "Document ownership could not be confirmed", ["One or more documents may belong to another applicant"], "Verify document ownership"
        elif missing_evidence:
            reason, uncertain, action = "Required evidence could not be extracted", missing_evidence, "Request clearer or replacement documents"
            status = ReviewStatus.REQUEST_MORE_INFORMATION
        elif state.reflection and not state.reflection.passed:
            reason, uncertain, action = "Quality-control checks found inconsistencies", state.reflection.issues, "Review contradictions and supporting evidence"
        else:
            reason = "Policy rules require referral"
            uncertain = state.policy_assessment.referred_rules_list if state.policy_assessment else []
            action = "Review referred policy rules"
        state.human_review = HumanReview(reason=reason, uncertain_checks=uncertain,
            policy_rules=state.policy_assessment.referred_rules_list if state.policy_assessment else [],
            relevant_evidence=[result.reason for result in state.policy_results if result.status != "PASS"],
            recommended_action=action, status=status)
        state.route = "HUMAN_REVIEW"
        state.workflow_status = WorkflowStatus.AWAITING_INFORMATION if status == ReviewStatus.REQUEST_MORE_INFORMATION else WorkflowStatus.AWAITING_REVIEW
        state.current_step = "Human Review"
        state.completed_steps.append("Human Review")
        state.check_results = build_check_results(state, self.required_documents)
        LOGGER.info("human_review_created", extra={"application_id": state.application_id,
            "review_reason": reason, "uncertain_check_count": len(uncertain), **state_summary(state)})
        return state.model_dump(mode="python")

    def _review_wait(self, value):
        state = self._state(value)
        submitted = interrupt({
            "application_id": state.application_id,
            "workflow_status": state.workflow_status.value,
            "review_status": state.human_review.status.value if state.human_review else None,
        })
        decision = HumanReviewDecision.model_validate(submitted)
        state.review_decision = decision
        if decision.decision == ReviewDecision.APPROVE:
            state.human_review.status = ReviewStatus.APPROVED_BY_REVIEWER
            state.workflow_status = WorkflowStatus.RUNNING
            state.route = "REVIEW_APPROVED"
        elif decision.decision == ReviewDecision.DECLINE:
            state.human_review.status = ReviewStatus.DECLINED_BY_REVIEWER
            state.workflow_status = WorkflowStatus.RUNNING
            state.route = "REVIEW_DECLINED"
        else:
            state.human_review.status = ReviewStatus.REQUEST_MORE_INFORMATION
            state.workflow_status = WorkflowStatus.AWAITING_INFORMATION
            state.route = "HUMAN_REVIEW"
        return state.model_dump(mode="python")

    def _review_decision_route(self, value):
        state = self._state(value)
        decision = state.review_decision.decision if state.review_decision else None
        route = {
            ReviewDecision.APPROVE: "underwriter",
            ReviewDecision.DECLINE: "decline",
            ReviewDecision.REQUEST_MORE_INFORMATION: "information",
        }.get(decision, "error")
        self._log_route(state, "Human Review", route, "review_decision")
        return route

    def _reviewer_decline(self, value):
        state = self._state(value)
        comment = state.review_decision.reviewer_comment if state.review_decision else None
        concerns = ["Declined by human reviewer"]
        if comment:
            concerns.append(comment)
        state.underwriting = UnderwritingDecision(decision="Decline", confidence=1.0,
            risk_rating="High", concerns=concerns, summary="Declined following human review.")
        state.route = "DECLINE"
        state.workflow_status = WorkflowStatus.DECLINED
        state.current_step = "Declined by Reviewer"
        state.completed_steps.append("Declined by Reviewer")
        state.check_results = build_check_results(state, self.required_documents)
        LOGGER.info("application_declined", extra={"application_id": state.application_id,
            "decline_source": "human_review", **state_summary(state)})
        return state.model_dump(mode="python")

    def _await_information(self, value):
        state = self._state(value)
        state.route = "HUMAN_REVIEW"
        state.workflow_status = WorkflowStatus.AWAITING_INFORMATION
        state.current_step = "Awaiting Information"
        return state.model_dump(mode="python")

    def _error_terminal(self, value):
        state = self._state(value)
        state.human_review = None
        state.route = "ERROR"
        state.workflow_status = WorkflowStatus.ERROR
        state.current_step = "Error"
        if "Error" not in state.completed_steps:
            state.completed_steps.append("Error")
        state.check_results = build_check_results(state, self.required_documents)
        LOGGER.error("workflow_error_terminal", extra={"application_id": state.application_id,
            "error_categories": [error.category.value for error in state.errors], **state_summary(state)})
        return state.model_dump(mode="python")

    def _decline(self, value):
        state = self._state(value)
        failures = state.policy_assessment.mandatory_failures if state.policy_assessment else []
        state.underwriting = UnderwritingDecision(decision="Decline", confidence=1.0, risk_rating="High", concerns=[f"Mandatory policy failure: {rule}" for rule in failures], summary="Declined due to mandatory policy failure.")
        state.route = "DECLINE"
        state.workflow_status = WorkflowStatus.DECLINED
        state.current_step = "Declined"
        state.completed_steps.append("Declined")
        state.check_results = build_check_results(state, self.required_documents)
        LOGGER.info("application_declined", extra={"application_id": state.application_id,
            "mandatory_failure_count": len(failures), **state_summary(state)})
        return state.model_dump(mode="python")

    def _complete(self, value):
        state = self._state(value)
        decision = state.underwriting.decision.upper() if state.underwriting else ""
        state.workflow_status = {
            "APPROVE": WorkflowStatus.APPROVED,
            "DECLINE": WorkflowStatus.DECLINED,
        }.get(decision, WorkflowStatus.COMPLETED)
        state.route = decision or "COMPLETED"
        state.current_step = "Completed"
        state.completed_steps.append("Completed")
        state.check_results = build_check_results(state, self.required_documents)
        return state.model_dump(mode="python")

    def build_graph(self, folder: Path):
        graph = StateGraph(MortgageState)
        graph.add_node("intake", self._node("Document Intake", ["document"], folder))
        graph.add_node("extraction", self._node("Extraction", ["passport", "employment", "payslip", "bank_statement"]))
        graph.add_node("ownership", self._node("Document Ownership", ["ownership"]))
        graph.add_node("identity_verification", self._node("Identity Verification", ["identity"]))
        graph.add_node("employment_verification", self._node("Employment Verification", ["employment_verification"]))
        graph.add_node("income_verification", self._node("Income Verification", ["income"]))
        graph.add_node("bank_verification", self._node("Bank Verification", ["bank"]))
        graph.add_node("fraud", self._node("Fraud Assessment", ["fraud"]))
        graph.add_node("affordability", self._node("Affordability", ["affordability"]))
        graph.add_node("policy_evaluation", self._node("Policy Evaluation", ["policy"]))
        graph.add_node("policy_summary", self._node("Policy Summary", ["policy_summary"]))
        graph.add_node("reflection", self._node("Quality Control", ["reflection"]))
        graph.add_node("underwriter", self._node("Final Underwriting", ["underwriter"]))
        graph.add_node("human_review", self._human_review)
        graph.add_node("review_wait", self._review_wait)
        graph.add_node("reviewer_decline", self._reviewer_decline)
        graph.add_node("await_information", self._await_information)
        graph.add_node("decline", self._decline)
        graph.add_node("error", self._error_terminal)
        graph.add_node("complete", self._complete)

        graph.add_edge(START, "intake")
        graph.add_conditional_edges("intake", self._intake_route, {"extract": "extraction", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("extraction", self._extraction_route, {"continue": "ownership", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("ownership", self._ownership_route, {"continue": "identity_verification", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("identity_verification", self._standard_route, {"continue": "employment_verification", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("employment_verification", self._standard_route, {"continue": "income_verification", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("income_verification", self._standard_route, {"continue": "bank_verification", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("bank_verification", self._standard_route, {"continue": "fraud", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("fraud", self._standard_route, {"continue": "affordability", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("affordability", self._standard_route, {"continue": "policy_evaluation", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("policy_evaluation", self._standard_route, {"continue": "policy_summary", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("policy_summary", self._policy_route, {"decline": "decline", "review": "human_review", "reflect": "reflection", "error": "error"})
        graph.add_conditional_edges("reflection", self._reflection_route, {"continue": "underwriter", "review": "human_review", "error": "error"})
        graph.add_conditional_edges("underwriter", self._standard_route, {"continue": "complete", "review": "human_review", "error": "error"})
        graph.add_edge("human_review", "review_wait")
        graph.add_conditional_edges("review_wait", self._review_decision_route,
            {"underwriter": "underwriter", "decline": "reviewer_decline",
             "information": "await_information", "error": "error"})
        for terminal in ("complete", "await_information", "reviewer_decline", "decline", "error"):
            graph.add_edge(terminal, END)
        return graph.compile(checkpointer=self.checkpointer)

    def run(self, folder, state=None):
        folder = Path(folder)
        started = perf_counter()
        initial = state or MortgageState()
        LOGGER.info("workflow_started", extra={"application_id": initial.application_id})
        self.graph = self.build_graph(folder)
        config = {"configurable": {"thread_id": initial.application_id}}
        invoked = self.graph.invoke(initial, config=config)
        if self.checkpointer is not None:
            invoked = self.graph.get_state(config).values
        result = self._state(invoked)
        LOGGER.info("workflow_completed", extra={"application_id": result.application_id,
            "duration_ms": round((perf_counter() - started) * 1000, 2), **state_summary(result)})
        return result
