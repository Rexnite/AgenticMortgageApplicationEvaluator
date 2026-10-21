from agents.base_agent import BaseAgent
from models.policy_assessment import PolicyAssessmentSummary


class PolicyAssessmentSummaryAgent(BaseAgent):
    name = "policy_summary"
    input_fields = ("policy_results",)
    output_fields = ("policy_assessment",)

    def evaluate(self, state):
        results = state.policy_results
        mandatory = [r.rule_id for r in results if r.status == "FAIL" and r.mandatory]
        referred = [r.rule_id for r in results if r.status == "REFER"]
        failed = [r.rule_id for r in results if r.status == "FAIL"]
        passed = [r.rule_id for r in results if r.status == "PASS"]
        overall = "FAIL" if mandatory else "REFER" if referred or failed else "PASS"
        return PolicyAssessmentSummary(total_rules=len(results), passed_rules=len(passed), failed_rules=len(failed), referred_rules=len(referred), overall_status=overall, mandatory_failures=mandatory, referred_rules_list=referred, failed_rules_list=failed, passed_rules_list=passed, summary=f"Policy outcome {overall}: {len(passed)} passed, {len(failed)} failed, {len(referred)} referred.", confidence=1.0)

    def run(self, state):
        state.current_step = "Policy Summary"
        state.policy_assessment = self.evaluate(state)
        state.completed_steps.append("Policy Summary")
        return state
