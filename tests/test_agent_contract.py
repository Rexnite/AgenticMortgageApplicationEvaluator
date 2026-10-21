import logging

import pytest

from agents.base_agent import AgentContractError, BaseAgent
from models.workflow import ErrorCategory
from workflow.graph import MortgageWorkflow
from workflow.state import MortgageState


class ContractAgent(BaseAgent):
    name = "contract_test"
    input_fields = ("passport",)
    output_fields = ("route",)

    def run(self, state):
        return "calculated"

    def update_state(self, state, result):
        state.route = result
        return state


def test_lifecycle_validates_runs_updates_and_logs(caplog):
    state = MortgageState(passport={})
    with caplog.at_level(logging.INFO, logger="mortgage.agent.contract_test"):
        result = ContractAgent(llm=object()).execute(state)
    assert result.route == "calculated"
    assert [record.msg for record in caplog.records] == ["agent_started", "agent_completed"]


def test_lifecycle_reports_declared_missing_input():
    with pytest.raises(AgentContractError) as exc_info:
        ContractAgent(llm=object()).execute(MortgageState())
    assert exc_info.value.agent == "contract_test"
    assert exc_info.value.field == "passport"
    assert exc_info.value.direction == "input"


def test_workflow_converts_contract_error_to_structured_missing_input():
    workflow = MortgageWorkflow.__new__(MortgageWorkflow)
    workflow.agents = {"contract": ContractAgent(llm=object())}
    result = workflow._node("Contract Stage", ["contract"])(MortgageState())
    error = result["errors"][0]
    assert error["category"] == ErrorCategory.MISSING_INPUT
    assert error["agent"] == "contract"
