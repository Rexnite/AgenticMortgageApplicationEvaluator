"""Durable local workflow service backed by LangGraph's official SQLite saver."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from models.workflow import (
    AuditEvent, HumanReviewDecision, ReviewDecision, WorkflowStatus,
)
from workflow.graph import MortgageWorkflow
from workflow.state import MortgageState


class WorkflowPersistenceError(RuntimeError):
    pass


class WorkflowNotFoundError(WorkflowPersistenceError):
    pass


class InvalidReviewDecisionError(WorkflowPersistenceError):
    pass


class PersistentMortgageWorkflow:
    """Service API for starting, inspecting, reviewing, and resuming workflows."""

    def __init__(self, db_path: str | Path | None = None, *, agents=None,
            required_documents=None, llm=None):
        self.db_path = Path(db_path or os.getenv("WORKFLOW_DB_PATH", ".data/workflows.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._audit_connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.workflow = MortgageWorkflow(agents=agents, required_documents=required_documents,
            checkpointer=self.checkpointer, llm=llm)
        # A placeholder folder is sufficient for checkpoint inspection and interrupt resume;
        # intake is not re-executed on review decisions.
        self.workflow.graph = self.workflow.build_graph(Path("."))
        self._setup_tables()

    @staticmethod
    def _config(application_id: str):
        return {"configurable": {"thread_id": application_id}}

    def _setup_tables(self):
        self._audit_connection.execute("PRAGMA journal_mode=WAL")
        self._audit_connection.execute(
            """CREATE TABLE IF NOT EXISTS workflow_registry (
                application_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )"""
        )
        self._audit_connection.execute(
            """CREATE TABLE IF NOT EXISTS workflow_audit (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                application_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )"""
        )
        self._audit_connection.commit()

    def _registered(self, application_id: str) -> bool:
        row = self._audit_connection.execute(
            "SELECT 1 FROM workflow_registry WHERE application_id = ?", (application_id,)
        ).fetchone()
        return row is not None

    def _require_registered(self, application_id: str):
        if not self._registered(application_id):
            raise WorkflowNotFoundError(f"Unknown application ID: {application_id}")

    def _audit(self, application_id: str, event_type: str, **metadata):
        timestamp = datetime.now(timezone.utc).isoformat()
        self._audit_connection.execute(
            "INSERT INTO workflow_audit(timestamp, application_id, event_type, metadata_json) VALUES (?, ?, ?, ?)",
            (timestamp, application_id, event_type,
             json.dumps(metadata, default=str, separators=(",", ":"))),
        )
        self._audit_connection.commit()

    def _audit_outcome(self, state: MortgageState):
        if self.workflow.required_documents.issubset(
                {document.document_type for document in state.documents}):
            self._audit(state.application_id, "DOCUMENTS_VALIDATED",
                document_type_count=len({document.document_type for document in state.documents}))
        if state.policy_assessment:
            self._audit(state.application_id, f"POLICY_{state.policy_assessment.overall_status}",
                policy_result_count=len(state.policy_results))
        event = {
            WorkflowStatus.AWAITING_REVIEW: "HUMAN_REVIEW_REQUESTED",
            WorkflowStatus.AWAITING_INFORMATION: "HUMAN_REVIEW_REQUESTED",
            WorkflowStatus.APPROVED: "APPROVED",
            WorkflowStatus.DECLINED: "DECLINED",
            WorkflowStatus.ERROR: "ERROR",
            WorkflowStatus.COMPLETED: "COMPLETED",
        }.get(state.workflow_status)
        if event:
            self._audit(state.application_id, event, workflow_status=state.workflow_status.value,
                route=state.route)

    def start(self, application_folder: str | Path, state: MortgageState | None = None) -> MortgageState:
        initial = state or MortgageState()
        if self._registered(initial.application_id):
            raise WorkflowPersistenceError(f"Application already exists: {initial.application_id}")
        timestamp = datetime.now(timezone.utc).isoformat()
        self._audit_connection.execute(
            "INSERT INTO workflow_registry(application_id, created_at) VALUES (?, ?)",
            (initial.application_id, timestamp),
        )
        self._audit_connection.commit()
        self._audit(initial.application_id, "WORKFLOW_STARTED")
        result = self.workflow.run(application_folder, initial)
        self._audit_outcome(result)
        return result

    def get_state(self, application_id: str) -> MortgageState:
        self._require_registered(application_id)
        snapshot = self.workflow.graph.get_state(self._config(application_id))
        if not snapshot.values:
            raise WorkflowNotFoundError(f"No checkpoint for application ID: {application_id}")
        return MortgageState.model_validate(snapshot.values)

    def get_status(self, application_id: str) -> WorkflowStatus:
        return self.get_state(application_id).workflow_status

    def submit_review(self, application_id: str,
            decision: HumanReviewDecision | dict) -> MortgageState:
        state = self.get_state(application_id)
        submitted = HumanReviewDecision.model_validate(decision)
        if state.workflow_status not in {
            WorkflowStatus.AWAITING_REVIEW, WorkflowStatus.AWAITING_INFORMATION,
        }:
            raise InvalidReviewDecisionError(
                f"Application is not awaiting review: {state.workflow_status.value}"
            )
        if state.policy_assessment and state.policy_assessment.mandatory_failures:
            raise InvalidReviewDecisionError("Mandatory policy failures cannot be overridden")
        if state.workflow_status == WorkflowStatus.AWAITING_INFORMATION and submitted.decision == ReviewDecision.APPROVE:
            raise InvalidReviewDecisionError("Missing information must be supplied before approval")
        self._audit(application_id, "REVIEW_DECISION_SUBMITTED",
            decision=submitted.decision.value, comment_supplied=bool(submitted.reviewer_comment))
        self._audit(application_id, "WORKFLOW_RESUMED", resume_source="human_review")
        self.workflow.graph.invoke(Command(resume=submitted.model_dump(mode="json")),
            config=self._config(application_id))
        result = self.get_state(application_id)
        self._audit_outcome(result)
        return result

    def resume(self, application_id: str, application_folder: str | Path) -> MortgageState:
        state = self.get_state(application_id)
        if state.workflow_status != WorkflowStatus.AWAITING_INFORMATION:
            raise WorkflowPersistenceError(
                f"Application is not awaiting information: {state.workflow_status.value}"
            )
        self._audit(application_id, "WORKFLOW_RESUMED", resume_source="updated_documents")
        # New evidence invalidates extraction and every downstream conclusion. A fresh state
        # with the same opaque ID safely recomputes the complete dependency chain.
        fresh = MortgageState(application_id=application_id)
        result = self.workflow.run(application_folder, fresh)
        self._audit_outcome(result)
        return result

    def get_history(self, application_id: str) -> list[AuditEvent]:
        self._require_registered(application_id)
        rows = self._audit_connection.execute(
            "SELECT timestamp, event_type, metadata_json FROM workflow_audit "
            "WHERE application_id = ? ORDER BY sequence", (application_id,),
        ).fetchall()
        return [AuditEvent(timestamp=timestamp, application_id=application_id,
            event_type=event_type, metadata=json.loads(metadata_json))
            for timestamp, event_type, metadata_json in rows]

    def close(self):
        self._audit_connection.close()
        self._checkpoint_connection.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
