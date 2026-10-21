import json

import pytest

from core.llm import (
    LLMClient, LocalLLMProvider, LocalModelUnavailable,
    StructuredOutputError,
)
from models.identity import IdentityVerification
from agents.identity import IdentityVerificationAgent
from models.passport import PassportData
from models.payslip import PayslipData
from models.bank_statement import BankStatementData
from workflow.state import MortgageState
from workflow.graph import MortgageWorkflow
from tests.test_workflow import fake_agents
from models.workflow import ErrorCategory


def test_default_provider_is_local_without_cloud_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("ALLOW_CLOUD_FALLBACK", raising=False)
    client = LLMClient()
    assert isinstance(client.provider, LocalLLMProvider)
    assert client.allow_cloud_fallback is False


def test_cloud_selection_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("ALLOW_CLOUD_FALLBACK", "false")
    with pytest.raises(ValueError):
        LLMClient()


def test_local_structured_output_success_and_one_repair(monkeypatch):
    provider = LocalLLMProvider()
    responses = iter(["not json", '{"verified": true}'])
    calls = []
    def fake_generate(_system, prompt, _model=None):
        calls.append(prompt)
        return next(responses)
    monkeypatch.setattr(provider, "generate", fake_generate)
    assert provider.generate_structured("", "verify")["verified"] is True
    assert len(calls) == 2
    assert "Return only valid JSON" in calls[-1]


def test_schema_failure_stops_after_one_retry(monkeypatch):
    provider = LocalLLMProvider()
    calls = []
    def fake_generate(_system, prompt, _model=None):
        calls.append(prompt)
        return json.dumps({"verified": "wrong"})
    monkeypatch.setattr(provider, "generate", fake_generate)
    with pytest.raises(StructuredOutputError):
        provider.generate_structured("", "verify", IdentityVerification)
    assert len(calls) == 2


class UnavailableProvider:
    def generate(self, *_args):
        raise LocalModelUnavailable("LOCAL_MODEL_UNAVAILABLE")
    def generate_structured(self, *_args):
        raise LocalModelUnavailable("LOCAL_MODEL_UNAVAILABLE")


class CloudSpy:
    def __init__(self): self.calls = 0
    def generate_structured(self, *_args):
        self.calls += 1
        return {"ok": True}


def test_cloud_not_called_when_fallback_disabled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("ALLOW_CLOUD_FALLBACK", "false")
    spy = CloudSpy()
    client = LLMClient(provider=UnavailableProvider(), cloud_provider=spy)
    with pytest.raises(LocalModelUnavailable):
        client.chat_json("", "test")
    assert spy.calls == 0


def test_cloud_fallback_requires_opt_in_and_uses_injected_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("ALLOW_CLOUD_FALLBACK", "true")
    spy = CloudSpy()
    client = LLMClient(provider=UnavailableProvider(), cloud_provider=spy)
    assert client.chat_json("", "test") == {"ok": True}
    assert spy.calls == 1


def test_unavailable_local_model_routes_to_review_with_controlled_code():
    agents = fake_agents()
    def fail(_state):
        raise LocalModelUnavailable("LOCAL_MODEL_UNAVAILABLE: test runtime absent")
    agents["passport"].function = fail
    state = MortgageWorkflow(agents=agents).run("unused")
    assert state.errors[0].code == "LOCAL_MODEL_UNAVAILABLE"
    assert state.errors[0].category == ErrorCategory.LLM_FAILURE
    assert state.human_review is not None


def test_local_model_health_failure_is_controlled(monkeypatch):
    provider = LocalLLMProvider(model="missing")
    monkeypatch.setattr(provider, "_request", lambda *_args: {"models": [{"name": "different"}]})
    with pytest.raises(LocalModelUnavailable, match="LOCAL_MODEL_UNAVAILABLE"):
        provider.health_check()


def test_exact_verification_does_not_invoke_provider():
    state = MortgageState(passport=PassportData(given_names="Jane", surname="Smith", extraction_complete=True),
        payslips=[PayslipData(employee_name="Jane Smith")],
        bank_statements=[BankStatementData(account_holder_name="Jane Smith")])
    agent = IdentityVerificationAgent(llm=LLMClient(provider=UnavailableProvider()))
    agent.execute(state)
    assert state.identity.verified


def test_ambiguous_verification_invokes_local_provider(monkeypatch):
    class LocalSpy:
        calls = 0
        def generate_structured(self, _system, _prompt, _model=None):
            self.calls += 1
            return IdentityVerification(applicant_name="Jane Smith", verified=False,
                inconsistencies=["Ambiguous"], missing_information=[], confidence=.5)
    monkeypatch.setenv("ALLOW_CLOUD_FALLBACK", "false")
    spy = LocalSpy()
    state = MortgageState(passport=PassportData(given_names="Jane", surname="Smith"),
        payslips=[PayslipData(employee_name="J. Smith")],
        bank_statements=[BankStatementData(account_holder_name="Jane Smith")])
    IdentityVerificationAgent(llm=LLMClient(provider=spy)).execute(state)
    assert spy.calls == 1
    assert state.identity.verified is False
