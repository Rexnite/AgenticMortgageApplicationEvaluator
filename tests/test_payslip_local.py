import json

import pytest

from agents.payslip import PayslipAgent, PayslipExtraction
from core.llm import LLMClient, LocalLLMProvider, StructuredOutputError


VALID = {"employer_name": "Example Ltd", "employee_name": "Jane Smith",
         "gross_pay": 3000.0, "net_pay": 2400.0}


def provider_with_responses(monkeypatch, responses):
    provider = LocalLLMProvider()
    payloads = []
    responses = iter(responses)

    def request(path, payload):
        payloads.append((path, payload))
        return {"message": {"content": next(responses)}}

    monkeypatch.setattr(provider, "_request", request)
    return provider, payloads


def test_payslip_uses_ollama_array_schema_and_marks_complete(monkeypatch):
    provider, payloads = provider_with_responses(monkeypatch, [json.dumps([VALID])])
    payslips = PayslipAgent(llm=LLMClient(provider=provider)).extract("sample OCR")
    assert len(payslips) == 1 and payslips[0].extraction_complete
    assert payloads[0][0] == "/api/chat"
    assert payloads[0][1]["format"]["type"] == "array"
    assert payloads[0][1]["format"]["$defs"]["PayslipExtraction"]["additionalProperties"] is False
    assert payloads[0][1]["options"] == {"temperature": 0}
    assert "pay_date" not in payloads[0][1]["messages"][0]["content"]


@pytest.mark.parametrize("malformed", [
    "not JSON",
    json.dumps(VALID),  # object instead of array
    json.dumps([{"employee_name": "Jane Smith", "gross_pay": 3000, "net_pay": 2400}]),
    json.dumps([{**VALID, "unexpected": "field"}]),
    json.dumps([{**VALID, "gross_pay": "not a number"}]),
])
def test_common_malformed_responses_retry_once_then_validate(monkeypatch, malformed):
    provider, payloads = provider_with_responses(monkeypatch,
        [malformed, json.dumps([VALID])])
    payslips = PayslipAgent(llm=LLMClient(provider=provider)).extract("sample OCR")
    assert payslips[0].extraction_complete
    assert len(payloads) == 2


def test_payslip_null_field_is_explicitly_incomplete(monkeypatch):
    provider, _ = provider_with_responses(monkeypatch,
        [json.dumps([{**VALID, "net_pay": None}])])
    payslips = PayslipAgent(llm=LLMClient(provider=provider)).extract("sample OCR")
    assert len(payslips) == 1 and not payslips[0].extraction_complete


def test_two_invalid_responses_fail_without_logging_ocr(monkeypatch, caplog):
    provider, payloads = provider_with_responses(monkeypatch,
        ['{"employee_name": "PRIVATE_OCR_VALUE"}',
         '{"employee_name": "PRIVATE_OCR_VALUE"}'])
    with caplog.at_level("WARNING"), pytest.raises(StructuredOutputError):
        PayslipAgent(llm=LLMClient(provider=provider)).extract("PRIVATE_OCR_VALUE")
    assert len(payloads) == 2
    assert "local_structured_output_retry" in caplog.text
    assert "local_structured_output_failed" in caplog.text
    assert "PRIVATE_OCR_VALUE" not in caplog.text


def test_payslip_model_requires_the_four_keys():
    assert set(PayslipExtraction.model_json_schema()["required"]) == {
        "employer_name", "employee_name", "gross_pay", "net_pay"}
