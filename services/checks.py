"""Read-only workflow check ledger; policy/routing decisions remain authoritative elsewhere."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from models.check_result import CheckResult, CheckStatus as S
from models.fraud import FraudSignalStatus
from services.calculations import annual_to_monthly, average_payslip_income, compare_income_sources, recurring_salary_credit
from services.comparisons import all_normalized_match


DOCUMENT_TYPES = ("Passport", "Employment Letter", "Payslip", "Bank Statement")
EXTRACTED_FIELDS = ("passport", "employment", "payslips", "bank_statements")
EXTRACTION_STEPS = ("Passport Extraction", "Employment Extraction", "Payslip Extraction", "Bank Statement Extraction")
EXTRACTION_AGENTS = ("passport", "employment", "payslip", "bank_statement")


def _row(stage, code, name, status, reason, **kwargs):
    return CheckResult(stage=stage, check_code=code, check_name=name,
                       status=status, reason=reason, **kwargs)


def _comparison(stage, code, name, outcome, *, source, action="Review discrepancy"):
    status = S.NOT_ASSESSED if outcome is None else S.PASS if outcome else S.REFER
    reason = ("Comparable evidence unavailable" if outcome is None else
              "Available evidence is consistent" if outcome else "Available evidence conflicts")
    return _row(stage, code, name, status, reason, source=source,
                action="Obtain evidence" if outcome is None else "Continue" if outcome else action)


def build_check_results(state, required_documents: Iterable[str] = DOCUMENT_TYPES) -> list[CheckResult]:
    """Regenerate a complete, deterministic snapshot; never carry stale rows forward."""
    rows: list[CheckResult] = []
    types = {document.document_type for document in state.documents}
    for document_type in DOCUMENT_TYPES:
        count = sum(document.document_type == document_type for document in state.documents)
        present = count > 0
        rows.append(_row("Document Intake", "DOC_" + document_type.upper().replace(" ", "_"),
            f"{document_type} present", S.PASS if present else S.FAIL if document_type in required_documents else S.NOT_ASSESSED,
            "Document type found" if present else "Required document type missing" if document_type in required_documents else "Optional document not supplied",
            document_type=document_type, metric=f"{count} file(s)", source="document classification",
            action="Continue" if present else "Request document"))

    for document_type, field, step, agent in zip(DOCUMENT_TYPES, EXTRACTED_FIELDS, EXTRACTION_STEPS, EXTRACTION_AGENTS):
        value = getattr(state, field)
        items = value if isinstance(value, list) else [value] if value is not None else []
        extracted = bool(items) and all(item.extraction_complete for item in items)
        present = document_type in types
        attempted = step in state.completed_steps or any(error.agent == agent for error in state.errors)
        status = (S.NOT_ASSESSED if not present or not attempted and not items else
                  S.PASS if extracted else S.FAIL)
        rows.append(_row("Extraction", "EXT_" + document_type.upper().replace(" ", "_"),
            f"{document_type} extraction", status,
            "Structured evidence extracted completely" if status == S.PASS else
            "Extraction not executed" if present and not attempted and not items else
            "Document present but extraction incomplete or failed" if status == S.FAIL else
            "Document unavailable",
            document_type=document_type, source="extraction agents",
            action="Continue" if status == S.PASS else "Request clearer evidence" if status == S.FAIL else "Await extraction"))

    quality = [doc for doc in state.documents if doc.extraction_success is False or
               (doc.extraction_confidence is not None and doc.extraction_confidence < .6)]
    quality_known = any(doc.extraction_success is not None for doc in state.documents)
    rows.append(_row("Extraction", "EXT_QUALITY", "Extraction quality",
        S.REFER if quality else S.PASS if quality_known else S.NOT_ASSESSED,
        "Extraction/readability issue recorded" if quality else "Recorded extraction quality acceptable" if quality_known else "Extraction quality metadata unavailable",
        metric=f"{len(quality)} issue(s)", source="extraction metadata",
        action="Review document quality" if quality else "Continue" if quality_known else "Obtain quality metadata"))

    ownership = state.document_ownership
    rows.append(_row("Ownership", "OWNERSHIP", "Document ownership consistency",
        S.REFER if state.rejected_documents or any(not item.belongs_to_applicant for item in ownership)
        else S.PASS if ownership else S.NOT_ASSESSED,
        "One or more documents rejected" if state.rejected_documents else "Ownership assessments consistent" if ownership else "Ownership not assessed",
        metric=f"{len(ownership)} assessed", source="ownership agent", action="Review ownership"))

    applicant = " ".join(filter(None, (getattr(state.passport, "given_names", None), getattr(state.passport, "surname", None))))
    names = [applicant, *(p.employee_name for p in state.payslips), *(b.account_holder_name for b in state.bank_statements)]
    rows.append(_comparison("Identity", "NAME_MATCH", "Name consistency", all_normalized_match(names), source="extracted names"))
    employers = [getattr(state.employment, "employer_name", None), *(p.employer_name for p in state.payslips)]
    rows.append(_comparison("Employment", "EMPLOYER_MATCH", "Employer consistency", all_normalized_match(employers), source="employment and payslips"))
    rows.append(_comparison("Income", "DECLARED_PAYSLIP", "Declared salary versus payslips",
        compare_income_sources(annual_to_monthly(getattr(state.employment, "annual_salary", None)), average_payslip_income(state.payslips, gross=True)), source="employment and payslips"))
    rows.append(_comparison("Income", "PAYSLIP_BANK", "Payslips versus salary credits",
        compare_income_sources(average_payslip_income(state.payslips), recurring_salary_credit(state.bank_statements)), source="payslips and bank statements"))
    holders = [b.account_holder_name for b in state.bank_statements]
    accounts = [b.account_number for b in state.bank_statements]
    rows.append(_comparison("Bank", "ACCOUNT_HOLDER", "Account-holder consistency",
        all_normalized_match(holders) if len(holders) > 1 else None, source="bank statements"))
    rows.append(_comparison("Bank", "ACCOUNT_NUMBER", "Account-number consistency",
        all_normalized_match(accounts) if len(accounts) > 1 else None, source="bank statements"))
    periods = [b.statement_months for b in state.bank_statements]
    rows.append(_row("Bank", "STATEMENT_PERIOD", "Statement period recorded",
        S.PASS if periods and all(p is not None and p > 0 for p in periods) else S.NOT_ASSESSED,
        "Positive period supplied" if periods and all(p is not None and p > 0 for p in periods) else "Statement periods unavailable",
        source="bank statements", action="Continue" if periods else "Request period evidence"))

    for stage, code, name, result in (
        ("Identity", "IDENTITY_VERIFICATION", "Identity verification", state.identity),
        ("Employment", "EMPLOYMENT_VERIFICATION", "Employment verification", state.employmentverification),
        ("Income", "INCOME_VERIFICATION", "Income verification", state.income),
        ("Bank", "BANK_VERIFICATION", "Bank verification", state.bank),
    ):
        status = S.NOT_ASSESSED if result is None else S.PASS if result.verified else S.REFER
        rows.append(_row(stage, code, name, status,
            "Specialist result unavailable" if result is None else "Specialist verified evidence" if result.verified else "Specialist could not verify evidence",
            source="verification agent", action="Continue" if status == S.PASS else "Human review"))

    signals = getattr(state.fraud, "signals", []) if state.fraud else []
    for signal in signals:
        status = {
            FraudSignalStatus.NOT_DETECTED: S.PASS,
            FraudSignalStatus.NOT_ASSESSED: S.NOT_ASSESSED,
            FraudSignalStatus.SUSPECTED: S.REFER,
            FraudSignalStatus.DETECTED: S.REFER,
        }[signal.status]
        rows.append(_row("Fraud", signal.code, signal.description, status,
            "Observed inconsistency or quality issue; not proof of fraud" if signal.status == FraudSignalStatus.DETECTED else
            "Concern requires review" if signal.status == FraudSignalStatus.SUSPECTED else
            "Check performed without finding" if status == S.PASS else "Check unavailable",
            source="structured fraud signal", action="Review signal" if status == S.REFER else "Continue" if status == S.PASS else "Do not infer a pass"))
    fraud_status = (S.NOT_ASSESSED if not signals else
        S.REFER if any(signal.status in {FraudSignalStatus.DETECTED, FraudSignalStatus.SUSPECTED} for signal in signals) else
        S.PASS if all(signal.status == FraudSignalStatus.NOT_DETECTED for signal in signals) else S.NOT_ASSESSED)
    rows.append(_row("Fraud", "FRAUD_ASSESSMENT", "Fraud assessment", fraud_status,
        "No structured signals available" if not signals else
        "One or more signals require review" if fraud_status == S.REFER else
        "All structured checks completed without findings" if fraud_status == S.PASS else
        "Some structured checks were not assessed",
        source="structured fraud signals", action="Human review" if fraud_status == S.REFER else
        "Continue" if fraud_status == S.PASS else "Obtain assessment"))

    affordability = state.affordability
    rows.append(_row("Affordability", "DTI", "Debt-to-income ratio",
        S.NOT_ASSESSED if not affordability or affordability.debt_to_income_ratio is None else
        S.PASS if affordability.debt_to_income_ratio <= .4 else S.REFER,
        "Metric unavailable" if not affordability or affordability.debt_to_income_ratio is None else "Compared with existing 40% policy threshold",
        metric=f"{affordability.debt_to_income_ratio:.1%}" if affordability and affordability.debt_to_income_ratio is not None else None,
        source="deterministic affordability", action="Review"))
    score = getattr(affordability, "affordability_score", None)
    rows.append(_row("Affordability", "AFFORDABILITY_SCORE", "Affordability score",
        S.NOT_ASSESSED if score is None else S.PASS if score >= 70 else S.REFER,
        "Score unavailable" if score is None else "Compared with existing score threshold",
        metric=str(score) if score is not None else None, source="deterministic affordability", action="Review"))

    for result in state.policy_results:
        safe_metric = (f"observed {result.actual_value}; target {result.expected_value}"
            if isinstance(result.actual_value, (int, float, bool)) and
               isinstance(result.expected_value, (int, float, bool)) else None)
        rows.append(_row("Policy", result.rule_id, result.name, S(result.status),
            "Required value missing" if result.actual_value is None else
            "Observed value meets rule" if result.status == "PASS" else
            "Observed value breaches mandatory rule" if result.status == "FAIL" else
            "Observed value triggers referral rule", metric=safe_metric,
            source="internal policy rule", action="Decline" if result.status == "FAIL" else "Human review" if result.status == "REFER" else "Continue"))
    rows.append(_row("Reflection", "QUALITY_CONTROL", "Quality-control assessment",
        S.NOT_ASSESSED if state.reflection is None else S.PASS if state.reflection.passed else S.REFER,
        "Quality-control pass not run" if state.reflection is None else "No contradiction found" if state.reflection.passed else "Contradiction or unsupported conclusion found",
        source="reflection agent", action="Human review" if state.reflection and not state.reflection.passed else "Continue"))
    return rows


@dataclass(frozen=True)
class CheckSummary:
    all_checks: list[CheckResult]

    @property
    def passed_checks(self): return [r for r in self.all_checks if r.status == S.PASS]
    @property
    def failed_checks(self): return [r for r in self.all_checks if r.status == S.FAIL]
    @property
    def referred_checks(self): return [r for r in self.all_checks if r.status == S.REFER]
    @property
    def unassessed_checks(self): return [r for r in self.all_checks if r.status == S.NOT_ASSESSED]
    @property
    def first_failure_check(self): return next((r for r in self.all_checks if r.status == S.FAIL), None)
    @property
    def first_failure_stage(self): return self.first_failure_check.stage if self.first_failure_check else None
    @property
    def failure_count(self): return len(self.failed_checks)
    @property
    def referral_count(self): return len(self.referred_checks)
    @property
    def unassessed_count(self): return len(self.unassessed_checks)


def render_check_table(checks: Iterable[CheckResult]) -> str:
    header = "| Stage | Check | Status | Reason | Metric | Action |"
    separator = "|---|---|---|---|---|---|"
    def safe(value): return str(value or "").replace("|", "/").replace("\n", " ")
    return "\n".join([header, separator, *(
        "| " + " | ".join(map(safe, (r.stage, r.check_name, r.status.value, r.reason, r.metric, r.action))) + " |"
        for r in checks)])
