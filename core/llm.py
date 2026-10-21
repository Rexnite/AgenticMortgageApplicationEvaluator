"""Local-first semantic model access; prompts are never logged."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pydantic import TypeAdapter, ValidationError

load_dotenv()
LOGGER = logging.getLogger("mortgage.llm")


class LocalModelUnavailable(RuntimeError):
    """Local runtime cannot be reached or lacks the configured model."""


class StructuredOutputError(ValueError):
    """Model output failed JSON/schema validation after one repair attempt."""


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
    def generate_structured(self, system_prompt: str, user_prompt: str,
                            response_model: Any = None) -> Any: ...


def _parse_json(raw: str, response_model: Any = None) -> Any:
    value = raw.strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines.pop()
        value = "\n".join(lines).strip()
    parsed = json.loads(value)
    return TypeAdapter(response_model).validate_python(parsed) if response_model is not None else parsed


class LocalLLMProvider:
    """Ollama /api/chat adapter using only the Python standard library."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 timeout: float = 120):
        self.model = model or os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
        self.base_url = (base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base_url + path, data=data,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LocalModelUnavailable("LOCAL_MODEL_UNAVAILABLE: Ollama is unreachable or returned an invalid response") from exc

    def health_check(self) -> bool:
        models = self._request("/api/tags").get("models", [])
        available = {item.get("name") or item.get("model") for item in models}
        if self.model not in available:
            raise LocalModelUnavailable("LOCAL_MODEL_UNAVAILABLE: configured Ollama model is not installed")
        return True

    def generate(self, system_prompt: str, user_prompt: str,
                 response_model: Any = None) -> str:
        output_format = (TypeAdapter(response_model).json_schema()
                         if response_model is not None else "json")
        response = self._request("/api/chat", {
            "model": self.model, "stream": False, "format": output_format,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
        })
        try:
            return response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LocalModelUnavailable("LOCAL_MODEL_UNAVAILABLE: Ollama response has no message content") from exc

    def generate_structured(self, system_prompt: str, user_prompt: str,
                            response_model: Any = None) -> Any:
        prompt = user_prompt
        for attempt in range(2):
            raw = self.generate(system_prompt, prompt, response_model)
            try:
                return _parse_json(raw, response_model)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                if attempt:
                    LOGGER.warning("local_structured_output_failed", extra={
                        "error_kind": type(exc).__name__, "attempt": attempt + 1,
                    })
                    raise StructuredOutputError("Local model returned invalid structured output after one retry") from None
                LOGGER.warning("local_structured_output_retry", extra={
                    "error_kind": type(exc).__name__, "attempt": attempt + 1,
                })
                prompt = user_prompt + "\nReturn only valid JSON matching the requested schema. No commentary or code fences."
        raise StructuredOutputError("Local model returned invalid structured output")


class MistralLLMProvider:
    """Explicit opt-in cloud fallback, never instantiated on the default path."""

    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is not configured")
        from mistralai import Mistral
        self.client = Mistral(api_key=key)
        self.model = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.complete(model=self.model, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        return response.choices[0].message.content

    def generate_structured(self, system_prompt: str, user_prompt: str,
                            response_model: Any = None) -> Any:
        return _parse_json(self.generate(system_prompt, user_prompt), response_model)


class LLMClient:
    """Compatibility facade for existing agents; local inference is the default."""

    def __init__(self, provider: LLMProvider | None = None,
                 cloud_provider: LLMProvider | None = None):
        selection = os.getenv("LLM_PROVIDER", "local").casefold()
        if selection not in {"local", "mistral"}:
            raise ValueError("LLM_PROVIDER must be 'local' or 'mistral'")
        if selection == "mistral" and os.getenv("ALLOW_CLOUD_FALLBACK", "false").casefold() != "true":
            raise ValueError("Cloud model use requires ALLOW_CLOUD_FALLBACK=true")
        self.provider = provider or (LocalLLMProvider() if selection == "local" else MistralLLMProvider())
        self.cloud_provider = cloud_provider
        self.allow_cloud_fallback = os.getenv("ALLOW_CLOUD_FALLBACK", "false").casefold() == "true"

    def _call(self, method: str, *args):
        try:
            return getattr(self.provider, method)(*args)
        except (LocalModelUnavailable, StructuredOutputError):
            if not self.allow_cloud_fallback or isinstance(self.provider, MistralLLMProvider):
                raise
            LOGGER.warning("cloud_fallback_used")
            cloud = self.cloud_provider or MistralLLMProvider()
            return getattr(cloud, method)(*args)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return self._call("generate", system_prompt, user_prompt)

    def chat_json(self, system_prompt: str, user_prompt: str,
                  response_model: Any = None) -> Any:
        return self._call("generate_structured", system_prompt, user_prompt, response_model)

    def health_check(self) -> bool:
        if isinstance(self.provider, LocalLLMProvider):
            return self.provider.health_check()
        return True
