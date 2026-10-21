from agents.base_agent import BaseAgent
from models.policy_engine import Operator, PolicyAssessment, PolicyResult, PolicyRule
from tools.rag import LocalPolicyRetriever, context_for_policy_results


DEFAULT_POLICY_RULES = [
    PolicyRule(rule_id="AGE001", name="Minimum Age", description="Applicant must be at least 18", state_path="passport.age", operator=Operator.GREATER_THAN_EQUAL, expected_value=18, mandatory=True),
    PolicyRule(rule_id="ID001", name="Identity Verification", description="Identity must be verified", state_path="identity.verified", operator=Operator.EQUAL, expected_value=True, mandatory=True),
    PolicyRule(rule_id="DTI001", name="Maximum Debt Ratio", description="Debt-to-income ratio must not exceed 40%", state_path="affordability.debt_to_income_ratio", operator=Operator.LESS_THAN_EQUAL, expected_value=0.40, on_fail="REFER", mandatory=False),
    PolicyRule(rule_id="AFF001", name="Minimum Affordability Score", description="Affordability score must be at least 70", state_path="affordability.affordability_score", operator=Operator.GREATER_THAN_EQUAL, expected_value=70, on_fail="REFER", mandatory=False),
    PolicyRule(rule_id="FRAUD001", name="Fraud Score", description="Fraud score must not exceed 60", state_path="fraud.fraud_score", operator=Operator.LESS_THAN_EQUAL, expected_value=60, on_fail="REFER", mandatory=False),
]


class PolicyAssessmentAgent(BaseAgent):
    name = "policy_engine"
    output_fields = ("policy_engine", "policy_results", "policy_context")

    def __init__(self, rules=None, llm=None, retriever=None):
        super().__init__(llm=llm)
        self.rules = list(rules or DEFAULT_POLICY_RULES)
        self.retriever = retriever or LocalPolicyRetriever.from_rules(self.rules)

    @staticmethod
    def get_value(obj, path):
        current = obj
        for field in path.split("."):
            if current is None:
                return None
            current = current.get(field) if isinstance(current, dict) else getattr(current, field, None)
        return current

    def evaluate(self, rule, state):
        actual = self.get_value(state, rule.state_path)
        if actual is None:
            return PolicyResult(rule_id=rule.rule_id, name=rule.name, actual_value=None, expected_value=rule.expected_value, status=rule.on_fail, reason="Required value is missing", mandatory=rule.mandatory)
        operations = {
            Operator.EQUAL: lambda a, b: a == b, Operator.NOT_EQUAL: lambda a, b: a != b,
            Operator.GREATER_THAN: lambda a, b: a > b, Operator.GREATER_THAN_EQUAL: lambda a, b: a >= b,
            Operator.LESS_THAN: lambda a, b: a < b, Operator.LESS_THAN_EQUAL: lambda a, b: a <= b,
            Operator.IN: lambda a, b: a in b, Operator.NOT_IN: lambda a, b: a not in b,
        }
        passed = operations[rule.operator](actual, rule.expected_value)
        return PolicyResult(rule_id=rule.rule_id, name=rule.name, actual_value=actual, expected_value=rule.expected_value, status="PASS" if passed else rule.on_fail, reason=f"{actual!r} {rule.operator.value} {rule.expected_value!r}", mandatory=rule.mandatory)

    def run(self, state):
        state.current_step = "Policy Evaluation"
        state.policy_rules = list(self.rules)
        state.policy_results = [self.evaluate(rule, state) for rule in state.policy_rules if rule.enabled]
        passed = sum(r.status == "PASS" for r in state.policy_results)
        failed = sum(r.status == "FAIL" for r in state.policy_results)
        referred = sum(r.status == "REFER" for r in state.policy_results)
        mandatory_failure = any(r.status == "FAIL" and r.mandatory for r in state.policy_results)
        overall = "FAIL" if mandatory_failure else "REFER" if referred or failed else "PASS"
        state.policy_engine = PolicyAssessment(passed=passed, failed=failed, referred=referred, overall_decision=overall, rules=state.policy_results, confidence=1.0)
        state.policy_context = context_for_policy_results(self.retriever, state.policy_results)
        state.completed_steps.append("Policy Evaluation")
        return state
