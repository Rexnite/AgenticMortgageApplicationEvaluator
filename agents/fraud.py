from __future__ import annotations

from statistics import mean

from agents.base_agent import BaseAgent
from models.fraud import (
    FraudAssessment, FraudSignal, FraudSynthesis, FraudSignalCategory as Category,
    FraudSignalSeverity as Severity, FraudSignalStatus as Status,
)
from services.calculations import (
    annual_to_monthly, average_payslip_income, compare_income_sources,
    recurring_salary_credit,
)
from services.comparisons import all_normalized_match


LIMITATIONS = [
    "No pixel-level manipulation or image-forgery analysis is performed.",
    "No signature, biometric, issuer-database, or government-document authentication is performed.",
    "PDF metadata is not treated as proof of authenticity.",
    "Transaction-level behavior cannot be assessed because transaction records are not available.",
]


class FraudAssessmentAgent(BaseAgent):
    name = "fraud_assessment"
    input_fields = ("passport", "employment", "payslips", "bank_statements")
    output_fields = ("fraud",)

    @staticmethod
    def _signal(code, category, status, severity, description, *, confidence=None,
            evidence_summary=None):
        return FraudSignal(code=code, category=category, status=status,
            severity=severity, confidence=confidence, description=description,
            evidence_summary=evidence_summary)

    def build_signals(self, state) -> list[FraudSignal]:
        signals = []

        # Specialist conclusions are authoritative evidence of observed contradictions,
        # but a failed verification is not itself a claim that fraud occurred.
        for code, category, verification, label in (
            ("IDENTITY_VERIFICATION_CONFLICT", Category.IDENTITY, state.identity, "Identity"),
            ("EMPLOYMENT_VERIFICATION_CONFLICT", Category.EMPLOYMENT, state.employmentverification, "Employment"),
            ("INCOME_VERIFICATION_CONFLICT", Category.INCOME, state.income, "Income"),
            ("BANK_VERIFICATION_CONFLICT", Category.BANKING, state.bank, "Bank account"),
        ):
            if verification is None:
                status, confidence, summary = Status.NOT_ASSESSED, None, "Specialist result unavailable"
            elif verification.verified is False or verification.inconsistencies:
                status, confidence, summary = Status.DETECTED, verification.confidence, "Specialist verification reported inconsistencies"
            else:
                status, confidence, summary = Status.NOT_DETECTED, verification.confidence, "Specialist verification found no inconsistency"
            signals.append(self._signal(code, category, status, Severity.HIGH if category == Category.IDENTITY else Severity.MEDIUM,
                f"{label} verification inconsistency", confidence=confidence, evidence_summary=summary))

        applicant = " ".join(filter(None, (
            getattr(state.passport, "given_names", None),
            getattr(state.passport, "surname", None),
        ))).strip()
        names = [applicant, *(p.employee_name for p in state.payslips),
            *(b.account_holder_name for b in state.bank_statements)]
        name_match = all_normalized_match(names)
        signals.append(self._comparison_signal("APPLICANT_NAME_CONSISTENCY", Category.CROSS_DOCUMENT,
            name_match, Severity.HIGH, "Applicant names are consistent across available documents"))

        employers = [getattr(state.employment, "employer_name", None), *(p.employer_name for p in state.payslips)]
        employer_match = all_normalized_match(employers)
        signals.append(self._comparison_signal("EMPLOYER_NAME_CONSISTENCY", Category.EMPLOYMENT,
            employer_match, Severity.MEDIUM, "Employer names are consistent across employment evidence"))

        net_income = average_payslip_income(state.payslips)
        salary_credit = recurring_salary_credit(state.bank_statements)
        income_match = compare_income_sources(net_income, salary_credit)
        signals.append(self._comparison_signal("PAYSLIP_BANK_INCOME_CONSISTENCY", Category.INCOME,
            income_match, Severity.MEDIUM, "Average net payslip income is consistent with salary credits"))

        declared_monthly = annual_to_monthly(getattr(state.employment, "annual_salary", None))
        gross_income = average_payslip_income(state.payslips, gross=True)
        declared_match = compare_income_sources(declared_monthly, gross_income)
        signals.append(self._comparison_signal("DECLARED_PAYSLIP_INCOME_CONSISTENCY", Category.INCOME,
            declared_match, Severity.MEDIUM, "Declared monthly gross income is consistent with gross payslip income"))

        holders = [statement.account_holder_name for statement in state.bank_statements]
        holder_match = all_normalized_match(holders) if len(holders) > 1 else None
        signals.append(self._comparison_signal("BANK_ACCOUNT_HOLDER_CONSISTENCY", Category.BANKING,
            holder_match, Severity.HIGH, "Account-holder names are consistent across bank statements"))

        accounts = [statement.account_number for statement in state.bank_statements]
        account_match = all_normalized_match(accounts) if len(accounts) > 1 else None
        signals.append(self._comparison_signal("BANK_ACCOUNT_NUMBER_CONSISTENCY", Category.BANKING,
            account_match, Severity.HIGH, "Account numbers are consistent across bank statements"))

        credits = [credit for statement in state.bank_statements for credit in statement.salary_credits]
        if len(credits) < 2 or any(credit <= 0 for credit in credits):
            variance_status, variance_summary = Status.NOT_ASSESSED, "At least two positive salary credits are required"
        else:
            relative_range = (max(credits) - min(credits)) / mean(credits)
            variance_status = Status.SUSPECTED if relative_range > 0.20 else Status.NOT_DETECTED
            variance_summary = "Salary-credit variation exceeds 20%" if variance_status == Status.SUSPECTED else "Salary-credit variation is within 20%"
        signals.append(self._signal("SALARY_CREDIT_VARIANCE", Category.BANKING, variance_status,
            Severity.MEDIUM, "Salary credits have a consistent recurring pattern",
            confidence=1.0 if variance_status != Status.NOT_ASSESSED else None,
            evidence_summary=variance_summary))

        document_signals = self._document_integrity_signals(state.documents)
        signals.extend(document_signals)
        return signals

    def _comparison_signal(self, code, category, comparison, severity, description):
        if comparison is None:
            status, summary, confidence = Status.NOT_ASSESSED, "Insufficient comparable evidence", None
        elif comparison:
            status, summary, confidence = Status.NOT_DETECTED, "Available normalized values match", 1.0
        else:
            status, summary, confidence = Status.DETECTED, "Available normalized or numeric values conflict", 1.0
        return self._signal(code, category, status, severity, description,
            confidence=confidence, evidence_summary=summary)

    def _document_integrity_signals(self, documents) -> list[FraudSignal]:
        if not documents:
            quality_status, quality_summary, confidence = Status.NOT_ASSESSED, "No document extraction records available", None
        else:
            unreadable = [document for document in documents
                if document.extraction_success is False or not document.extracted_text.strip()]
            low_confidence = [document for document in documents
                if document.extraction_confidence is not None and document.extraction_confidence < 0.60]
            invalid_pages = [document for document in documents
                if document.page_count is not None and document.page_count <= 0]
            if unreadable or low_confidence or invalid_pages:
                quality_status, quality_summary = Status.DETECTED, "One or more documents have an observable extraction or readability issue"
            else:
                assessed = all(document.extraction_success is not None for document in documents)
                quality_status = Status.NOT_DETECTED if assessed else Status.NOT_ASSESSED
                quality_summary = "Recorded extraction checks found no quality issue" if assessed else "Extraction quality metadata is incomplete"
            confidence = 1.0 if quality_status != Status.NOT_ASSESSED else None
        return [
            self._signal("DOCUMENT_EXTRACTION_QUALITY", Category.DOCUMENT_INTEGRITY,
                quality_status, Severity.LOW, "Document extraction quality and readability",
                confidence=confidence, evidence_summary=quality_summary),
            self._signal("DOCUMENT_TAMPERING_FORENSICS", Category.DOCUMENT_INTEGRITY,
                Status.NOT_ASSESSED, Severity.HIGH, "Pixel-level or structural tampering analysis",
                evidence_summary="No forensic tampering detector is implemented"),
            self._signal("DUPLICATE_PAGE_ANALYSIS", Category.DOCUMENT_INTEGRITY,
                Status.NOT_ASSESSED, Severity.MEDIUM, "Duplicate-page analysis",
                evidence_summary="Per-page hashes or images are not retained"),
            self._signal("RECURRING_OVERDRAFT_USAGE", Category.BANKING,
                Status.NOT_ASSESSED, Severity.MEDIUM, "Recurring overdraft usage",
                evidence_summary="Only overdraft limits are extracted; usage history is unavailable"),
        ]

    @staticmethod
    def score_signals(signals: list[FraudSignal]) -> int:
        weights = {
            (Status.DETECTED, Severity.HIGH): 30,
            (Status.DETECTED, Severity.MEDIUM): 15,
            (Status.DETECTED, Severity.LOW): 5,
            (Status.SUSPECTED, Severity.HIGH): 20,
            (Status.SUSPECTED, Severity.MEDIUM): 10,
            (Status.SUSPECTED, Severity.LOW): 3,
        }
        # Use only the strongest signal per category to avoid double-counting correlated checks.
        category_scores = {}
        for signal in signals:
            value = weights.get((signal.status, signal.severity), 0)
            category_scores[signal.category] = max(category_scores.get(signal.category, 0), value)
        return min(100, sum(category_scores.values()))

    def _semantic_synthesis(self, state, signals):
        active = [signal.model_dump(mode="json") for signal in signals
            if signal.status in {Status.DETECTED, Status.SUSPECTED}]
        prompt = f"""
            You are reviewing structured mortgage fraud-risk signals.
            Do not create new signals or factual evidence.
            Do not claim fraud has been detected unless a supplied signal supports that wording.
            Distinguish observed inconsistency, suspicion, and unavailable checks.
            Do not invent image-forensic, signature, metadata, or transaction findings.

            Active structured signals: {active}
            Specialist summary:
            identity_verified={getattr(state.identity, 'verified', None)}
            employment_verified={getattr(state.employmentverification, 'verified', None)}
            income_verified={getattr(state.income, 'verified', None)}
            bank_verified={getattr(state.bank, 'verified', None)}

            Return JSON with exactly:
            {{"suspected_concerns": [], "recommended_action": "", "confidence": 0.0}}
        """
        return self.chat_json("", prompt, FraudSynthesis)

    def assess(self, state, *, use_llm=True) -> FraudAssessment:
        signals = self.build_signals(state)
        score = self.score_signals(signals)
        risk = "High" if score >= 70 else "Medium" if score >= 40 else "Low"
        detected = [signal.description for signal in signals if signal.status == Status.DETECTED]
        suspected = [signal.description for signal in signals if signal.status == Status.SUSPECTED]
        unassessed = [signal.description for signal in signals if signal.status == Status.NOT_ASSESSED]
        recommendation = "Refer for specialist fraud review" if score >= 40 else "Continue standard underwriting with stated limitations"
        confidence = sum(signal.status != Status.NOT_ASSESSED for signal in signals) / len(signals)
        if use_llm and (detected or suspected):
            synthesis = self._semantic_synthesis(state, signals)
            suspected.extend(str(item) for item in synthesis.get("suspected_concerns", []))
            recommendation = synthesis.get("recommended_action") or recommendation
            confidence = min(confidence, float(synthesis.get("confidence", confidence)))
        applicant_name = getattr(state.identity, "applicant_name", "") or ""
        return FraudAssessment(applicant_name=applicant_name, fraud_score=score, risk=risk,
            indicators=detected + suspected, confidence=round(confidence, 3), signals=signals,
            confirmed_inconsistencies=detected, suspected_concerns=suspected,
            unassessed_checks=unassessed, limitations=LIMITATIONS,
            recommended_action=recommendation)

    def extract(self, state) -> FraudAssessment:
        return self.assess(state)

    def run(self, state):
        state.current_step = "Fraud Assessment"
        state.fraud = self.assess(state)
        state.completed_steps.append("Fraud Assessment")
        self.logger.info("fraud_assessment_summary", extra={
            "fraud_score": state.fraud.fraud_score,
            "fraud_risk": state.fraud.risk,
            "signal_count": len(state.fraud.signals),
            "unassessed_signal_count": len(state.fraud.unassessed_checks),
        })
        return state
