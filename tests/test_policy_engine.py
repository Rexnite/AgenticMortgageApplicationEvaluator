import pytest

from agents.policy_assessment import PolicyAssessmentSummaryAgent
from agents.policy_engine import PolicyAssessmentAgent
from models.affordability import AffordabilityData
from models.fraud import FraudAssessment
from models.identity import IdentityVerification
from models.passport import PassportData
from models.policy_engine import PolicyResult
from workflow.state import MortgageState


def state_for(age=30, identity=True, fraud=10, dti=0.3, score=80):
    return MortgageState(
        passport=PassportData(age=age),
        identity=IdentityVerification(applicant_name="Applicant", verified=identity, inconsistencies=[], missing_information=[], confidence=1),
        fraud=FraudAssessment(applicant_name="Applicant", fraud_score=fraud, risk="Low", indicators=[], confidence=1),
        affordability=AffordabilityData(debt_to_income_ratio=dti, affordability_score=score),
    )


def evaluate(state):
    PolicyAssessmentAgent().run(state)
    PolicyAssessmentSummaryAgent().run(state)
    return state


@pytest.mark.parametrize("age,expected", [(17, "FAIL"), (18, "PASS"), (40, "PASS")])
def test_age_policy(age, expected):
    result = next(r for r in evaluate(state_for(age=age)).policy_results if r.rule_id == "AGE001")
    assert result.status == expected


def test_identity_failure_and_multiple_mandatory_failures_force_fail():
    state = evaluate(state_for(age=17, identity=False))
    assert state.policy_engine.overall_decision == "FAIL"
    assert state.policy_assessment.overall_status == "FAIL"
    assert set(state.policy_assessment.mandatory_failures) == {"AGE001", "ID001"}
    assert all(isinstance(result, PolicyResult) for result in state.policy_results)


def test_policy_results_store_evaluations_not_append_return_values():
    state = evaluate(state_for())
    assert len(state.policy_results) == 5
    assert all(isinstance(result, PolicyResult) for result in state.policy_results)
    assert state.policy_engine.rules == state.policy_results


@pytest.mark.parametrize("kwargs,rule", [({"fraud": 80}, "FRAUD001"), ({"dti": .5}, "DTI001"), ({"score": 60}, "AFF001")])
def test_non_mandatory_thresholds_refer(kwargs, rule):
    state = evaluate(state_for(**kwargs))
    assert next(r for r in state.policy_results if r.rule_id == rule).status == "REFER"
    assert state.policy_assessment.overall_status == "REFER"


def test_valid_application_passes():
    state = evaluate(state_for())
    assert state.policy_engine.rules == state.policy_results
    assert state.policy_engine.overall_decision == "PASS"
    assert state.policy_assessment.overall_status == "PASS"
