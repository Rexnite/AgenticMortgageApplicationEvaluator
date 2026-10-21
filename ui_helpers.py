"""Pure presentation helpers for the Streamlit demo UI."""

from __future__ import annotations

from pathlib import Path
import json
import io
from typing import Any

import fitz
from PIL import Image, UnidentifiedImageError

from models.check_result import CheckStatus
from models.workflow import ReviewDecision, WorkflowStatus
from services.checks import CheckSummary
from services.workflow_persistence import (
    InvalidReviewDecisionError, WorkflowNotFoundError,
)


STAGES = (
    ("Document Intake", ("Document Scan",)),
    ("Extraction", ("Passport Extraction", "Employment Extraction", "Payslip Extraction", "Bank Statement Extraction")),
    ("Ownership", ("Document Ownership",)),
    ("Identity", ("Identity Verification",)),
    ("Employment", ("Employment Verification",)),
    ("Income", ("Income Verification",)),
    ("Bank", ("Bank Account Verification",)),
    ("Fraud", ("Fraud Assessment",)),
    ("Affordability", ("Affordability Assessment",)),
    ("Policy", ("Policy Evaluation", "Policy Summary")),
    ("Reflection", ("Quality Control",)),
    ("Underwriting", ("Underwriting Decision",)),
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_UPLOAD_FILES = 20
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}


class UploadValidationError(ValueError):
    """An uploaded document is unsafe, unsupported, or unreadable."""


def validate_uploads(uploaded_files) -> list[tuple[str, bytes]]:
    files = list(uploaded_files or [])
    if not files:
        raise UploadValidationError("Upload at least one document.")
    if len(files) > MAX_UPLOAD_FILES:
        raise UploadValidationError(f"Upload no more than {MAX_UPLOAD_FILES} documents at once.")
    validated = []
    for uploaded in files:
        display_name = Path(uploaded.name).name
        suffix = Path(display_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise UploadValidationError(f"{display_name}: unsupported file type.")
        data = uploaded.getvalue()
        if not data:
            raise UploadValidationError(f"{display_name}: file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadValidationError(f"{display_name}: exceeds the 15 MB per-file limit.")
        try:
            if suffix == ".pdf":
                document = fitz.open(stream=data, filetype="pdf")
                try:
                    if document.needs_pass:
                        raise UploadValidationError(f"{display_name}: password-protected PDFs are not supported.")
                    if document.page_count < 1:
                        raise UploadValidationError(f"{display_name}: PDF contains no pages.")
                finally:
                    document.close()
            else:
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
        except UploadValidationError:
            raise
        except (fitz.FileDataError, UnidentifiedImageError, OSError, ValueError):
            raise UploadValidationError(f"{display_name}: file is corrupt or does not match its extension.") from None
        validated.append((suffix, data))
    return validated


def result_export(state) -> str:
    """Serialize assessed results without local paths or raw extracted document text."""
    payload = state.model_dump(mode="json", exclude={
        "documents": {"__all__": {"file_path", "extracted_text"}},
    })
    payload["documents"] = [
        {key: value for key, value in item.items() if key not in {"file_path", "extracted_text"}}
        for item in payload.get("documents", [])
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)

STATUS_LABELS = {
    CheckStatus.PASS: "✅ PASS",
    CheckStatus.FAIL: "❌ FAIL",
    CheckStatus.REFER: "⚠️ REFER",
    CheckStatus.NOT_ASSESSED: "➖ NOT_ASSESSED",
}


def sample_application_path(root: str | Path | None = None) -> Path | None:
    candidate = Path(root or Path(__file__).resolve().parent) / "sample_application"
    return candidate if candidate.is_dir() else None


def state_summary(state) -> dict[str, Any]:
    checks = CheckSummary(state.check_results)
    return {
        "workflow_status": state.workflow_status.value,
        "route": state.route,
        "application_id": state.application_id,
        "documents": len(state.documents),
        "passed": len(checks.passed_checks),
        "failed": checks.failure_count,
        "referred": checks.referral_count,
        "not_assessed": checks.unassessed_count,
        "fraud_risk": getattr(state.fraud, "risk", None),
        "fraud_score": getattr(state.fraud, "fraud_score", None),
        "policy_status": getattr(state.policy_assessment, "overall_status", None),
        "decision": getattr(state.underwriting, "decision", None),
    }


def check_table_rows(state, status: str = "All", stage: str = "All") -> list[dict[str, Any]]:
    rows = state.check_results
    if status != "All":
        rows = [row for row in rows if row.status.value == status]
    if stage != "All":
        rows = [row for row in rows if row.stage == stage]
    return [{
        "Stage": row.stage, "Check": row.check_name,
        "Status": STATUS_LABELS[row.status], "Reason": row.reason,
        "Metric": row.metric or "", "Action": row.action or "",
    } for row in rows]


def workflow_stage_rows(state) -> list[dict[str, str]]:
    completed = set(state.completed_steps)
    failed_stage = state.errors[-1].stage if state.errors else None
    result = []
    for label, backend_steps in STAGES:
        if all(step in completed for step in backend_steps):
            status = "Completed"
        elif failed_stage and (failed_stage == label or failed_stage.startswith(label)):
            status = "Failed"
        elif any(step == state.current_step for step in backend_steps) or state.current_step == label:
            status = "Current"
        else:
            status = "Not Run"
        result.append({"Stage": label, "Status": status})
    if state.workflow_status in {WorkflowStatus.AWAITING_REVIEW, WorkflowStatus.AWAITING_INFORMATION}:
        result.append({"Stage": "Human Review", "Status": "Human Review"})
    return result


def review_action_allowed(state, decision: ReviewDecision | str) -> tuple[bool, str | None]:
    decision = ReviewDecision(decision)
    if state.workflow_status not in {WorkflowStatus.AWAITING_REVIEW, WorkflowStatus.AWAITING_INFORMATION}:
        return False, "Application is not awaiting review."
    if state.policy_assessment and state.policy_assessment.mandatory_failures:
        return False, "Mandatory policy failures cannot be overridden."
    if state.workflow_status == WorkflowStatus.AWAITING_INFORMATION and decision == ReviewDecision.APPROVE:
        return False, "Additional information must be supplied before approval."
    return True, None


def load_application(service, application_id: str):
    try:
        return service.get_state(application_id.strip()), None
    except WorkflowNotFoundError:
        return None, "Application ID was not found."
    except (ValueError, InvalidReviewDecisionError):
        return None, "Application state could not be loaded."


def structured_fields(model, *, exclude: set[str] | None = None) -> dict[str, Any]:
    if model is None:
        return {}
    values = model.model_dump(mode="json")
    return {key.replace("_", " ").title(): value for key, value in values.items()
            if key not in (exclude or set())}
