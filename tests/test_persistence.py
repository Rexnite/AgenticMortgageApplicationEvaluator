from pathlib import Path

import pytest

from models.workflow import HumanReviewDecision, WorkflowStatus
from services.workflow_persistence import (
    InvalidReviewDecisionError, PersistentMortgageWorkflow,
    WorkflowNotFoundError,
)
from tests.test_workflow import fake_agents


def service(db_path, **agent_options):
    return PersistentMortgageWorkflow(db_path, agents=fake_agents(**agent_options))


def test_start_creates_durable_checkpoint(tmp_path):
    db = tmp_path / "workflows.db"
    with service(db) as workflows:
        state = workflows.start("unused")
        recovered = workflows.get_state(state.application_id)
        assert db.exists()
        assert recovered.application_id == state.application_id
        assert recovered.workflow_status == WorkflowStatus.APPROVED


def test_new_service_instance_recovers_same_application(tmp_path):
    db = tmp_path / "workflows.db"
    first = service(db, fraud=80)
    started = first.start("unused")
    application_id = started.application_id
    first.close()

    with service(db, fraud=80) as second:
        recovered = second.get_state(application_id)
        assert recovered.application_id == application_id
        assert recovered.workflow_status == WorkflowStatus.AWAITING_REVIEW
        assert recovered.human_review.reason == "Policy rules require referral"


def test_reviewer_approve_resumes_to_final_underwriter(tmp_path):
    calls = []
    db = tmp_path / "workflows.db"
    first = service(db, fraud=80, calls=calls)
    state = first.start("unused")
    application_id = state.application_id
    assert state.workflow_status == WorkflowStatus.AWAITING_REVIEW
    assert calls == []
    first.close()

    with service(db, fraud=80, calls=calls) as workflows:
        resumed = workflows.submit_review(application_id,
            HumanReviewDecision(decision="APPROVE", reviewer_comment="Reviewed evidence"))
        assert resumed.workflow_status == WorkflowStatus.APPROVED
        assert resumed.underwriting.decision == "Approve"
        assert calls == ["underwriter"]


def test_reviewer_decline_uses_deterministic_terminal(tmp_path):
    calls = []
    with service(tmp_path / "workflows.db", fraud=80, calls=calls) as workflows:
        state = workflows.start("unused")
        resumed = workflows.submit_review(state.application_id,
            {"decision": "DECLINE", "reviewer_comment": "Risk not acceptable"})
        assert resumed.workflow_status == WorkflowStatus.DECLINED
        assert resumed.underwriting.decision == "Decline"
        assert "Declined by human reviewer" in resumed.underwriting.concerns
        assert calls == []


def test_missing_documents_await_information_and_can_recompute(tmp_path):
    db = tmp_path / "workflows.db"
    first = service(db, complete=False)
    state = first.start("unused")
    application_id = state.application_id
    assert state.workflow_status == WorkflowStatus.AWAITING_INFORMATION
    with pytest.raises(InvalidReviewDecisionError):
        first.submit_review(application_id, {"decision": "APPROVE"})
    first.close()

    with service(db, complete=True) as second:
        resumed = second.resume(application_id, "unused")
        assert resumed.application_id == application_id
        assert resumed.workflow_status == WorkflowStatus.APPROVED


def test_mandatory_failure_cannot_be_reviewed_or_overridden(tmp_path):
    with service(tmp_path / "workflows.db", age=17) as workflows:
        state = workflows.start("unused")
        assert state.workflow_status == WorkflowStatus.DECLINED
        with pytest.raises(InvalidReviewDecisionError):
            workflows.submit_review(state.application_id, {"decision": "APPROVE"})


def test_audit_history_is_ordered_and_safe(tmp_path):
    with service(tmp_path / "workflows.db", fraud=80) as workflows:
        state = workflows.start("unused")
        workflows.submit_review(state.application_id, {"decision": "DECLINE"})
        history = workflows.get_history(state.application_id)
        event_types = [event.event_type for event in history]
        assert event_types[0] == "WORKFLOW_STARTED"
        assert "POLICY_REFER" in event_types
        assert "HUMAN_REVIEW_REQUESTED" in event_types
        assert event_types.index("REVIEW_DECISION_SUBMITTED") < event_types.index("WORKFLOW_RESUMED")
        assert event_types[-1] == "DECLINED"
        serialized = " ".join(str(event.metadata) for event in history)
        assert "Risk not acceptable" not in serialized


def test_application_checkpoints_are_isolated(tmp_path):
    with service(tmp_path / "workflows.db") as workflows:
        first = workflows.start("unused")
        second = workflows.start("unused")
        assert first.application_id != second.application_id
        assert workflows.get_state(first.application_id).application_id == first.application_id
        assert workflows.get_state(second.application_id).application_id == second.application_id


def test_unknown_application_id_is_controlled(tmp_path):
    with service(tmp_path / "workflows.db") as workflows:
        with pytest.raises(WorkflowNotFoundError):
            workflows.get_state("unknown-id")
        with pytest.raises(WorkflowNotFoundError):
            workflows.get_history("unknown-id")
