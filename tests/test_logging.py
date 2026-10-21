import json
import logging

from core.logging import JsonFormatter, state_summary
from workflow.state import MortgageState


def test_json_formatter_includes_structured_context_and_exception():
    formatter = JsonFormatter()
    try:
        raise ValueError("safe failure")
    except ValueError:
        record = logging.getLogger("test").makeRecord(
            "test", logging.ERROR, __file__, 1, "stage_failed", (),
            exc_info=__import__("sys").exc_info(), extra={"stage": "Policy"},
        )
    payload = json.loads(formatter.format(record))
    assert payload["event"] == "stage_failed"
    assert payload["stage"] == "Policy"
    assert payload["exception_type"] == "ValueError"


def test_state_summary_is_useful_without_serializing_sensitive_models():
    state = MortgageState(route="HUMAN_REVIEW")
    summary = state_summary(state)
    assert summary["route"] == "HUMAN_REVIEW"
    assert summary["document_count"] == 0
    assert "passport" not in summary
    assert "bank_statements" not in summary
