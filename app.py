"""Local Streamlit demonstration UI over the existing mortgage workflow service."""

from __future__ import annotations

import logging
import os
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4

import streamlit as st

from core.llm import (
    GeminiLLMProvider, LLMClient, LocalLLMProvider, LocalModelUnavailable,
    MistralLLMProvider,
)
from core.logging import configure_logging
from models.workflow import HumanReviewDecision, ReviewDecision, WorkflowStatus
from services.workflow_persistence import (
    InvalidReviewDecisionError, PersistentMortgageWorkflow, WorkflowPersistenceError,
)
from ui_helpers import (
    check_table_rows, load_application, review_action_allowed,
    result_export, state_summary, structured_fields, validate_uploads,
    UploadValidationError, workflow_stage_rows,
)


LOGGER = logging.getLogger("mortgage.ui")

def configure_secrets() -> None:
    """Map supported Streamlit secrets to environment variables without logging values."""
    for key in ("LLM_PROVIDER", "MISTRAL_API_KEY", "MISTRAL_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL",
                "ALLOW_CLOUD_FALLBACK", "LOCAL_LLM_MODEL", "LOCAL_LLM_BASE_URL"):
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value is not None and key not in os.environ:
            os.environ[key] = str(value)


def session_workspace() -> Path:
    if "workspace" not in st.session_state:
        token = uuid4().hex
        folder = Path(tempfile.gettempdir()) / "mortgage-evaluator" / token
        folder.mkdir(parents=True, exist_ok=False)
        st.session_state.workspace = str(folder)
    return Path(st.session_state.workspace)


def service():
    api_key = st.session_state.get("user_provider_api_key", "").strip()
    provider_name = st.session_state.get("user_provider", "Gemini")
    if api_key:
        provider = GeminiLLMProvider(api_key=api_key) if provider_name == "Gemini" else MistralLLMProvider(api_key=api_key)
        llm = LLMClient(provider=provider)
    else:
        llm = None
    return PersistentMortgageWorkflow(db_path=session_workspace() / "workflows.db", llm=llm)


def save_uploads(uploaded_files, folder: Path) -> None:
    for suffix, data in validate_uploads(uploaded_files):
        (folder / f"document-{uuid4().hex}{suffix}").write_bytes(data)


def run_from_folder(folder: Path):
    with service() as workflow:
        return workflow.start(folder)


def resume_from_folder(application_id: str, folder: Path):
    with service() as workflow:
        return workflow.resume(application_id, folder)


def provider_panel() -> bool:
    provider_name = st.sidebar.selectbox(
        "Model provider", ["Gemini", "Mistral"], key="user_provider",
        help="Choose the cloud provider associated with your API key.")
    user_key = st.sidebar.text_input(
        f"Your {provider_name} API key",
        type="password",
        key="user_provider_api_key",
        help="Used only for this browser session. It is not written to the workflow database, logs, or repository.",
        placeholder="Enter a key to enable underwriting",
    ).strip()
    if user_key:
        model = (os.getenv("GEMINI_MODEL", "gemini-2.5-flash") if provider_name == "Gemini"
                 else os.getenv("MISTRAL_MODEL", "mistral-medium-latest"))
        st.sidebar.write(f"**Provider:** {provider_name} (your session key)")
        st.sidebar.write(f"**Configured model:** {model}")
        st.sidebar.write("**Server:** Ready to authenticate on submission")
        st.sidebar.warning(f"Uploaded document text will be sent to {provider_name}. Usage and quota apply to the key you entered.")
        return True

    selection = os.getenv("LLM_PROVIDER", "local").casefold()
    fallback = os.getenv("ALLOW_CLOUD_FALLBACK", "false").casefold() == "true"
    st.sidebar.write(f"**Provider:** {'Ollama' if selection == 'local' else selection.title()}")
    model = (os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b") if selection == "local"
             else os.getenv("MISTRAL_MODEL", "mistral-medium-latest"))
    st.sidebar.write(f"**Configured model:** {model}")
    try:
        client = LLMClient(provider=LocalLLMProvider(timeout=2)) if selection == "local" else LLMClient()
        available = client.health_check()
    except (LocalModelUnavailable, RuntimeError, ValueError):
        available = False
    st.sidebar.write(f"**Server:** {'Available' if available else 'Unavailable'}")
    st.sidebar.write(f"**Cloud fallback:** {'Enabled' if fallback else 'Disabled'}")
    if not available and selection == "local":
        st.sidebar.warning("Ollama or the configured model is unavailable. Semantic stages will refer safely.")
    if selection == "mistral":
        st.sidebar.warning("Uploaded document text is sent to the configured Mistral cloud API for semantic processing.")
    return available


def render_metrics(state) -> None:
    summary = state_summary(state)
    first = st.columns(4)
    first[0].metric("Workflow Status", summary["workflow_status"])
    first[1].metric("Final Route", summary["route"])
    first[2].metric("Documents", summary["documents"])
    first[3].metric("Application ID", summary["application_id"][:8] + "…")
    second = st.columns(4)
    second[0].metric("Checks Passed", summary["passed"])
    second[1].metric("Checks Failed", summary["failed"])
    second[2].metric("Checks Referred", summary["referred"])
    second[3].metric("Not Assessed", summary["not_assessed"])
    st.caption(f"Application ID: `{summary['application_id']}`")


def render_extracted(state) -> None:
    st.subheader("Extracted Information")
    passport, employment, payslips, banks = st.tabs(
        ["Passport", "Employment", "Payslips", "Bank Statements"])
    with passport:
        if state.passport:
            st.json(structured_fields(state.passport), expanded=True)
            if not state.passport.extraction_complete:
                st.warning("Incomplete extraction")
        else:
            st.info("Not extracted")
    with employment:
        if state.employment:
            st.json(structured_fields(state.employment), expanded=True)
            if not state.employment.extraction_complete:
                st.warning("Incomplete extraction")
        else:
            st.info("Not extracted")
    with payslips:
        if not state.payslips:
            st.info("No payslips extracted")
        for index, item in enumerate(state.payslips, 1):
            with st.expander(f"Payslip {index}", expanded=index == 1):
                st.json(structured_fields(item), expanded=True)
                if not item.extraction_complete:
                    st.warning("Incomplete extraction")
    with banks:
        if not state.bank_statements:
            st.info("No bank statements extracted")
        for index, item in enumerate(state.bank_statements, 1):
            with st.expander(f"Bank statement {index}", expanded=index == 1):
                st.json(structured_fields(item), expanded=True)
                if not item.extraction_complete:
                    st.warning("Incomplete extraction")


def render_checks(state) -> None:
    st.subheader("Application Checks")
    left, right = st.columns(2)
    status = left.selectbox("Status", ["All", "FAIL", "REFER", "NOT_ASSESSED", "PASS"])
    stages = ["All", *sorted({row.stage for row in state.check_results})]
    stage = right.selectbox("Stage", stages)
    rows = check_table_rows(state, status, stage)
    st.dataframe(rows, width="stretch", hide_index=True,
        column_order=["Stage", "Check", "Status", "Reason", "Metric", "Action"])


def render_risk_tabs(state) -> None:
    verification, fraud, affordability, policy, reflection = st.tabs(
        ["Verification", "Fraud", "Affordability", "Policy", "Reflection"])
    with verification:
        for title, result in (("Identity", state.identity),
                ("Employment", state.employmentverification),
                ("Income", state.income), ("Bank", state.bank)):
            with st.expander(title, expanded=True):
                st.json(structured_fields(result)) if result else st.info("Not assessed")
    with fraud:
        if not state.fraud:
            st.info("Not assessed")
        else:
            st.metric("Fraud score", state.fraud.fraud_score)
            st.write(f"**Risk band:** {state.fraud.risk}")
            counts = {name: sum(signal.status.value == name for signal in state.fraud.signals)
                      for name in ("DETECTED", "SUSPECTED", "NOT_ASSESSED")}
            st.caption("Signals indicate review concerns and do not by themselves prove fraud.")
            st.write(counts)
            st.dataframe([{"Code": signal.code, "Category": signal.category.value,
                "Status": signal.status.value, "Severity": signal.severity.value,
                "Description": signal.description} for signal in state.fraud.signals],
                width="stretch", hide_index=True)
    with affordability:
        st.json(structured_fields(state.affordability)) if state.affordability else st.info("Not assessed")
    with policy:
        if state.policy_assessment:
            st.write(f"**Overall status:** {state.policy_assessment.overall_status}")
            st.write(state.policy_assessment.summary)
        st.dataframe([{"Rule": item.rule_id, "Name": item.name, "Status": item.status,
            "Reason": item.reason} for item in state.policy_results],
            width="stretch", hide_index=True)
    with reflection:
        st.json(structured_fields(state.reflection)) if state.reflection else st.info("Not run")


def render_review(state) -> None:
    if state.workflow_status not in {WorkflowStatus.AWAITING_REVIEW, WorkflowStatus.AWAITING_INFORMATION}:
        return
    st.subheader("Human Review")
    review = state.human_review
    if review:
        st.warning(review.reason)
        st.write("**Uncertain checks:**", review.uncertain_checks or "None recorded")
        st.write("**Relevant evidence:**", review.relevant_evidence or "None recorded")
        st.write("**Relevant policy rules:**", review.policy_rules or "None recorded")
        st.write("**Recommended action:**", review.recommended_action)
    if state.workflow_status == WorkflowStatus.AWAITING_INFORMATION:
        st.info("Additional documents or evidence are required. Upload a complete replacement set below.")
        additions = st.file_uploader("Replacement application documents", type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True, key="additional_documents")
        if st.button("Resume with Updated Evidence", disabled=not additions):
            try:
                with tempfile.TemporaryDirectory(prefix="mortgage-ui-") as temporary:
                    folder = Path(temporary)
                    save_uploads(additions, folder)
                    st.session_state.state = resume_from_folder(state.application_id, folder)
                st.rerun()
            except Exception:
                LOGGER.exception("ui_updated_evidence_failed", extra={"application_id": state.application_id})
                st.error("The updated evidence could not be processed. Check the local services and try again.")
        return
    decision = st.selectbox("Review decision", [item.value for item in ReviewDecision])
    comment = st.text_area("Reviewer comment (not shown in audit history)")
    allowed, reason = review_action_allowed(state, decision)
    if reason:
        st.warning(reason)
    if st.button("Submit Review Decision", disabled=not allowed):
        try:
            with service() as workflow:
                st.session_state.state = workflow.submit_review(state.application_id,
                    HumanReviewDecision(decision=decision, reviewer_comment=comment or None))
            st.rerun()
        except (InvalidReviewDecisionError, WorkflowPersistenceError) as exc:
            LOGGER.warning("ui_review_rejected", extra={"application_id": state.application_id,
                "error_kind": type(exc).__name__})
            st.error("That review action is not allowed for the current application state.")


def render_history(state) -> None:
    with st.expander("Workflow History"):
        try:
            with service() as workflow:
                history = workflow.get_history(state.application_id)
            st.dataframe([{"Timestamp": item.timestamp, "Event": item.event_type,
                "Status / Route": item.metadata.get("workflow_status") or item.metadata.get("route") or ""}
                for item in history], width="stretch", hide_index=True)
        except WorkflowPersistenceError:
            st.info("No persisted history is available.")


def render_state(state) -> None:
    render_metrics(state)
    for error in state.errors:
        st.error(f"{error.stage} — {error.message}")
    if state.workflow_status in {WorkflowStatus.AWAITING_REVIEW, WorkflowStatus.AWAITING_INFORMATION}:
        st.warning("The workflow detected incomplete or inconsistent evidence and referred the case for review.")
    st.subheader("Workflow Progress")
    st.dataframe(workflow_stage_rows(state), width="stretch", hide_index=True)
    render_checks(state)
    render_extracted(state)
    render_risk_tabs(state)
    render_review(state)
    render_history(state)
    st.download_button("Download results (JSON)", data=result_export(state),
        file_name=f"mortgage-result-{state.application_id}.json",
        mime="application/json", use_container_width=True)


def reset_application() -> None:
    workspace = st.session_state.get("workspace")
    if workspace:
        shutil.rmtree(workspace, ignore_errors=True)
    for key in ("workspace", "state", "uploads", "is_running", "user_provider", "user_provider_api_key"):
        st.session_state.pop(key, None)


def main() -> None:
    st.set_page_config(page_title="Mortgage Underwriting Demo", page_icon="🏠", layout="wide")
    configure_secrets()
    configure_logging()
    st.title("Mortgage Underwriting Evaluator")
    st.caption("Prototype for mentor evaluation only — not a production lending decision service.")
    provider_available = provider_panel()

    if st.sidebar.button("New application / reset"):
        reset_application()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Resume Existing Application")
    application_id = st.sidebar.text_input("Application ID")
    if st.sidebar.button("Load", disabled=not application_id.strip()):
        with service() as workflow:
            loaded, error = load_application(workflow, application_id)
        if error:
            st.sidebar.error(error)
        else:
            st.session_state.state = loaded
            st.rerun()

    st.subheader("Application Documents")
    st.write("Upload a passport, employment letter, payslip(s), and bank statement(s). "
             "Accepted formats: PDF, PNG, JPG/JPEG; maximum 15 MB each and 20 files per run.")
    st.caption("Use synthetic data for this hosted prototype. Community Cloud storage is temporary and is not a records system.")
    if not provider_available:
        st.warning("Choose Gemini or Mistral and enter your own API key in the sidebar to enable Run Underwriting. The key stays in this browser session and is not persisted by the app.")
    uploads = st.file_uploader("Upload applicant documents", type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True, key="uploads")

    if st.button("Run Underwriting", type="primary",
            disabled=(not uploads or st.session_state.get("is_running", False) or not provider_available)):
        st.session_state.is_running = True
        try:
            with st.status("Evaluating documents…", expanded=True) as status:
                st.write("Validating uploads and creating an isolated application workspace")
                with tempfile.TemporaryDirectory(prefix="uploads-", dir=session_workspace()) as temporary:
                    temporary_folder = Path(temporary)
                    save_uploads(uploads, temporary_folder)
                    st.write("Running extraction, verification, affordability and policy workflow")
                    state = run_from_folder(temporary_folder)
                st.session_state.state = state
                status.update(label="Evaluation complete", state="complete")
            st.rerun()
        except UploadValidationError as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("ui_workflow_failed")
            st.error("The workflow could not be completed. Verify the configured model provider and try again.")
        finally:
            st.session_state.is_running = False

    if state := st.session_state.get("state"):
        st.divider()
        render_state(state)


if __name__ == "__main__":
    main()
