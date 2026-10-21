from models.check_result import CheckStatus as S
from models.fraud import FraudAssessment, FraudSignal, FraudSignalCategory, FraudSignalSeverity, FraudSignalStatus
from services.checks import CheckSummary, render_check_table
from services.workflow_persistence import PersistentMortgageWorkflow
from tests.test_workflow import fake_agents
from workflow.graph import MortgageWorkflow
from workflow.state import MortgageState
from services.checks import build_check_results
from models.document import Document
from models.payslip import PayslipData
from pathlib import Path


def get(checks, code):
    return next(row for row in checks if row.check_code == code)


def test_happy_path_has_pass_and_unassessed_rows():
    state = MortgageWorkflow(agents=fake_agents()).run("unused")
    checks = state.check_results
    assert get(checks, "DOC_PASSPORT").status == S.PASS
    assert get(checks, "EXT_PASSPORT").status == S.PASS
    assert get(checks, "IDENTITY_VERIFICATION").status == S.PASS
    assert get(checks, "AGE001").status == S.PASS
    assert get(checks, "QUALITY_CONTROL").status == S.PASS
    assert get(checks, "NAME_MATCH").status == S.NOT_ASSESSED
    summary = CheckSummary(checks)
    assert summary.passed_checks and summary.unassessed_checks
    assert summary.failure_count == 0
    assert "| Stage | Check | Status |" in render_check_table(checks)


def test_missing_document_and_failed_extraction_are_distinct():
    missing = MortgageWorkflow(agents=fake_agents(complete=False)).run("unused")
    assert get(missing.check_results, "DOC_BANK_STATEMENT").status == S.FAIL
    assert get(missing.check_results, "EXT_BANK_STATEMENT").status == S.NOT_ASSESSED
    failed = MortgageWorkflow(agents=fake_agents(missing_extraction="bank_statement")).run("unused")
    assert get(failed.check_results, "DOC_BANK_STATEMENT").status == S.PASS
    assert get(failed.check_results, "EXT_BANK_STATEMENT").status == S.FAIL


def test_extraction_never_run_is_unassessed_but_incomplete_result_fails():
    document = Document(file_name="payslip.pdf", file_path=Path("payslip.pdf"),
        extension=".pdf", size=1, document_type="Payslip")
    state = MortgageState(documents=[document])
    assert get(build_check_results(state), "EXT_PAYSLIP").status == S.NOT_ASSESSED
    state.completed_steps.append("Payslip Extraction")
    assert get(build_check_results(state), "EXT_PAYSLIP").status == S.FAIL
    state.payslips = [PayslipData(employee_name="Jane", gross_pay=3000, net_pay=2400,
        extraction_complete=False)]
    assert get(build_check_results(state), "EXT_PAYSLIP").status == S.FAIL
    state.payslips[0].extraction_complete = True
    assert get(build_check_results(state), "EXT_PAYSLIP").status == S.PASS


def test_verification_referral_policy_and_first_failure():
    state = MortgageWorkflow(agents=fake_agents(identity=False)).run("unused")
    assert get(state.check_results, "IDENTITY_VERIFICATION").status == S.REFER
    assert get(state.check_results, "ID001").status == S.FAIL
    summary = CheckSummary(state.check_results)
    assert summary.first_failure_check.check_code == "ID001"
    assert summary.first_failure_stage == "Policy"
    referred = MortgageWorkflow(agents=fake_agents(fraud=80)).run("unused")
    assert get(referred.check_results, "FRAUD001").status == S.REFER


def test_employment_income_and_fraud_unassessed_rows():
    state = MortgageWorkflow(agents=fake_agents()).run("unused")
    assert get(state.check_results, "EMPLOYER_MATCH").status == S.NOT_ASSESSED
    assert get(state.check_results, "DECLARED_PAYSLIP").status == S.PASS
    assert get(state.check_results, "PAYSLIP_BANK").status == S.PASS
    # Older injected fraud fixtures have no typed signals; no clean checks are fabricated.
    assert not [row for row in state.check_results if row.source == "structured fraud signal"]
    assert get(state.check_results, "FRAUD_ASSESSMENT").status == S.NOT_ASSESSED


def test_fraud_signal_statuses_preserve_unassessed_and_do_not_confirm_fraud():
    signals = [
        FraudSignal(code=code, category=FraudSignalCategory.IDENTITY,
            status=status, severity=FraudSignalSeverity.MEDIUM, description=code)
        for code, status in (
            ("SIGNAL_CLEAR", FraudSignalStatus.NOT_DETECTED),
            ("SIGNAL_UNKNOWN", FraudSignalStatus.NOT_ASSESSED),
            ("SIGNAL_CONCERN", FraudSignalStatus.SUSPECTED),
            ("SIGNAL_INCONSISTENCY", FraudSignalStatus.DETECTED),
        )
    ]
    state = MortgageState(fraud=FraudAssessment(applicant_name="Applicant", fraud_score=40,
        risk="Medium", confidence=.8, signals=signals))
    checks = build_check_results(state)
    assert [get(checks, code).status for code in
        ("SIGNAL_CLEAR", "SIGNAL_UNKNOWN", "SIGNAL_CONCERN", "SIGNAL_INCONSISTENCY")] == [
            S.PASS, S.NOT_ASSESSED, S.REFER, S.REFER]
    assert get(checks, "FRAUD_ASSESSMENT").status == S.REFER


def test_employment_referral_and_quality_control_contradiction():
    agents = fake_agents(reflection_passed=False)
    original = agents["employment_verification"].function
    def unverified(state):
        state = original(state)
        state.employmentverification.verified = False
        return state
    agents["employment_verification"].function = unverified
    state = MortgageWorkflow(agents=agents).run("unused")
    assert get(state.check_results, "EMPLOYMENT_VERIFICATION").status == S.REFER
    assert get(state.check_results, "QUALITY_CONTROL").status == S.REFER


def test_check_ledger_survives_checkpoint_and_recomputes_after_new_evidence(tmp_path):
    db = tmp_path / "checks.db"
    first = PersistentMortgageWorkflow(db, agents=fake_agents(complete=False))
    missing = first.start("unused")
    application_id = missing.application_id
    first.close()
    second = PersistentMortgageWorkflow(db, agents=fake_agents(complete=True))
    try:
        recovered = second.get_state(application_id)
        assert get(recovered.check_results, "DOC_BANK_STATEMENT").status == S.FAIL
        updated = second.resume(application_id, "unused")
        assert get(updated.check_results, "DOC_BANK_STATEMENT").status == S.PASS
        assert get(second.get_state(application_id).check_results, "DOC_BANK_STATEMENT").status == S.PASS
    finally:
        second.close()
