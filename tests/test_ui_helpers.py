from pathlib import Path

from models.check_result import CheckResult, CheckStatus
from models.policy_assessment import PolicyAssessmentSummary
from models.workflow import ReviewDecision, WorkflowStatus
from services.workflow_persistence import WorkflowNotFoundError
from ui_helpers import (
    check_table_rows, load_application, review_action_allowed,
    sample_application_path, state_summary, structured_fields, workflow_stage_rows,
)
from workflow.state import MortgageState


def test_state_to_summary_handles_incomplete_state():
    state = MortgageState(check_results=[
        CheckResult(stage="Intake", check_code="A", check_name="A",
            status=CheckStatus.PASS, reason="ok"),
        CheckResult(stage="Extraction", check_code="B", check_name="B",
            status=CheckStatus.NOT_ASSESSED, reason="not run"),
    ])
    summary = state_summary(state)
    assert summary["documents"] == 0
    assert summary["passed"] == 1
    assert summary["not_assessed"] == 1
    assert summary["fraud_score"] is None
    assert structured_fields(None) == {}


def test_check_table_rows_maps_and_filters_status():
    state = MortgageState(check_results=[
        CheckResult(stage="Policy", check_code="A", check_name="Age",
            status=CheckStatus.FAIL, reason="below minimum", metric="17", action="Decline"),
        CheckResult(stage="Identity", check_code="B", check_name="Name",
            status=CheckStatus.PASS, reason="consistent"),
    ])
    rows = check_table_rows(state, "FAIL", "Policy")
    assert rows == [{"Stage": "Policy", "Check": "Age", "Status": "❌ FAIL",
        "Reason": "below minimum", "Metric": "17", "Action": "Decline"}]


def test_stage_mapping_does_not_invent_completion():
    state = MortgageState(completed_steps=["Document Scan"], current_step="Passport Extraction")
    rows = {item["Stage"]: item["Status"] for item in workflow_stage_rows(state)}
    assert rows["Document Intake"] == "Completed"
    assert rows["Extraction"] == "Current"
    assert rows["Identity"] == "Not Run"


def test_review_action_guards_status_and_mandatory_failures():
    state = MortgageState()
    assert review_action_allowed(state, ReviewDecision.APPROVE)[0] is False
    state.workflow_status = WorkflowStatus.AWAITING_INFORMATION
    assert review_action_allowed(state, ReviewDecision.APPROVE)[0] is False
    assert review_action_allowed(state, ReviewDecision.REQUEST_MORE_INFORMATION)[0] is True
    state.workflow_status = WorkflowStatus.AWAITING_REVIEW
    state.policy_assessment = PolicyAssessmentSummary(total_rules=1, passed_rules=0,
        failed_rules=1, referred_rules=0, overall_status="FAIL",
        mandatory_failures=["AGE001"], referred_rules_list=[], failed_rules_list=["AGE001"],
        passed_rules_list=[], summary="failed", confidence=1)
    assert review_action_allowed(state, ReviewDecision.APPROVE) == (
        False, "Mandatory policy failures cannot be overridden.")


def test_unknown_application_is_user_friendly():
    class MissingService:
        def get_state(self, _application_id):
            raise WorkflowNotFoundError("private detail")
    state, error = load_application(MissingService(), "unknown")
    assert state is None
    assert error == "Application ID was not found."
    assert "private" not in error


def test_sample_application_path_detection(tmp_path):
    assert sample_application_path(tmp_path) is None
    (tmp_path / "sample_application").mkdir()
    assert sample_application_path(tmp_path) == Path(tmp_path) / "sample_application"
