"""Structured, privacy-safe logging for the underwriting workflow."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from models.check_result import CheckStatus


STANDARD_FIELDS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line for easy ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in STANDARD_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
            payload["exception"] = str(record.exc_info[1])
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(level: str | None = None, *, stream=None) -> None:
    """Configure application logging once, with LOG_LEVEL as an override."""
    selected = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, selected, logging.INFO)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)


def state_summary(state: Any) -> dict[str, Any]:
    """Return useful operational counts/statuses while excluding applicant PII."""
    policy = getattr(state, "policy_assessment", None)
    reflection = getattr(state, "reflection", None)
    underwriting = getattr(state, "underwriting", None)
    review = getattr(state, "human_review", None)
    fraud = getattr(state, "fraud", None)
    checks = getattr(state, "check_results", []) or []
    return {
        "document_count": len(getattr(state, "documents", []) or []),
        "payslip_count": len(getattr(state, "payslips", []) or []),
        "bank_statement_count": len(getattr(state, "bank_statements", []) or []),
        "rejected_document_count": len(getattr(state, "rejected_documents", []) or []),
        "policy_result_count": len(getattr(state, "policy_results", []) or []),
        "policy_context_count": len(getattr(state, "policy_context", []) or []),
        "error_count": len(getattr(state, "errors", []) or []),
        "completed_step_count": len(getattr(state, "completed_steps", []) or []),
        "route": getattr(state, "route", None),
        "policy_status": getattr(policy, "overall_status", None),
        "reflection_passed": getattr(reflection, "passed", None),
        "decision": getattr(underwriting, "decision", None),
        "fraud_score": getattr(fraud, "fraud_score", None),
        "fraud_risk": getattr(fraud, "risk", None),
        "fraud_signal_count": len(getattr(fraud, "signals", []) or []),
        "fraud_unassessed_count": len(getattr(fraud, "unassessed_checks", []) or []),
        "check_count": len(checks),
        "check_pass_count": sum(item.status == CheckStatus.PASS for item in checks),
        "check_fail_count": sum(item.status == CheckStatus.FAIL for item in checks),
        "check_refer_count": sum(item.status == CheckStatus.REFER for item in checks),
        "check_unassessed_count": sum(item.status == CheckStatus.NOT_ASSESSED for item in checks),
        "review_status": getattr(review, "status", None),
        "workflow_status": getattr(state, "workflow_status", None),
    }
