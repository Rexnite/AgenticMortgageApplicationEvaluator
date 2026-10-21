from streamlit.testing.v1 import AppTest
from pathlib import Path

from models.check_result import CheckResult, CheckStatus
from workflow.state import MortgageState


def test_streamlit_entrypoint_renders_workflow_state(monkeypatch):
    # Avoid an external Ollama health request; no model call is needed to render state.
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("ALLOW_CLOUD_FALLBACK", "false")
    state = MortgageState(route="HUMAN_REVIEW", check_results=[
        CheckResult(stage="Extraction", check_code="EXT_PASSPORT",
            check_name="Passport extraction", status=CheckStatus.NOT_ASSESSED,
            reason="Extraction not executed", action="Await extraction")
    ])
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15)
    app.session_state["state"] = state
    app.run()
    assert not app.exception
    assert [item.value for item in app.title] == ["Mortgage Underwriting Evaluator"]
    assert "Application Checks" in [item.value for item in app.subheader]
    assert any(metric.label == "Final Route" and metric.value == "HUMAN_REVIEW"
               for metric in app.metric)
    assert "Bundled sample application" not in [item.value for item in app.radio]
