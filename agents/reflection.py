from agents.base_agent import BaseAgent
from models.fraud import FraudSignalStatus
from models.workflow import ReflectionResult


class ReflectionAgent(BaseAgent):
    """One deterministic quality-control pass over specialist conclusions."""
    name = "reflection"
    input_fields = ("policy_assessment", "policy_context")
    output_fields = ("reflection",)

    def assess(self, state):
        missing, contradictions, policy_issues, confidence_issues = [], [], [], []
        specialists = {"identity": state.identity, "employment": state.employmentverification,
            "income": state.income, "bank": state.bank, "fraud": state.fraud,
            "affordability": state.affordability}
        missing.extend(name for name, value in specialists.items() if value is None)
        for name, value in specialists.items():
            confidence = getattr(value, "confidence", None) if value else None
            if confidence is not None and confidence < 0.6:
                confidence_issues.append(f"{name} confidence is {confidence:.2f}")
        if state.income and state.affordability:
            reported, calculated = state.income.monthly_income, state.affordability.average_monthly_net_income
            if reported and calculated and abs(reported - calculated) / max(abs(reported), 1) > 0.15:
                contradictions.append("Income verification and calculated monthly net income differ by more than 15%")
        if state.policy_engine and state.policy_assessment and state.policy_engine.overall_decision != state.policy_assessment.overall_status:
            policy_issues.append("Policy engine and policy summary outcomes do not match")
        if state.policy_assessment and state.policy_assessment.overall_status == "PASS":
            failed = [name for name in ("identity", "employment", "income", "bank") if getattr(specialists.get(name), "verified", True) is False]
            if failed:
                policy_issues.append(f"Policy passed despite failed verification: {', '.join(failed)}")
        if state.fraud:
            active_fraud_signals = [signal for signal in state.fraud.signals
                if signal.status in {FraudSignalStatus.DETECTED, FraudSignalStatus.SUSPECTED}]
            if (state.fraud.fraud_score >= 40 or state.fraud.indicators) and not active_fraud_signals:
                policy_issues.append("Fraud conclusion is not supported by structured active signals")
            fraud_rule = next((result for result in state.policy_results if result.rule_id == "FRAUD001"), None)
            if fraud_rule and fraud_rule.actual_value != state.fraud.fraud_score:
                policy_issues.append("Fraud policy result does not use the current fraud score")
        issues = missing + contradictions + policy_issues + confidence_issues
        return ReflectionResult(passed=not issues, issues=issues, missing_evidence=missing,
            contradictions=contradictions, policy_inconsistencies=policy_issues,
            confidence_concerns=confidence_issues,
            evidence={
                "policy_outcome": getattr(state.policy_assessment, "overall_status", None),
                "policy_context": [
                    {"id": item.id, "title": item.title, "source": item.source}
                    for item in state.policy_context
                ],
                "fraud_signals": {
                    "detected": sum(signal.status == FraudSignalStatus.DETECTED for signal in state.fraud.signals) if state.fraud else 0,
                    "suspected": sum(signal.status == FraudSignalStatus.SUSPECTED for signal in state.fraud.signals) if state.fraud else 0,
                    "not_assessed": sum(signal.status == FraudSignalStatus.NOT_ASSESSED for signal in state.fraud.signals) if state.fraud else 0,
                },
            })

    def run(self, state):
        state.current_step = "Quality Control"
        state.reflection = self.assess(state)
        state.completed_steps.append("Quality Control")
        return state
