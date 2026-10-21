"""Local-first semantic model access; prompts are never logged."""

from __future__ import annotations

import json
import logging
import os
import re
import time
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


class ModelProviderError(RuntimeError):
    """A cloud model request failed with a safe, user-facing message."""


class ProviderRateLimitError(ModelProviderError):
    """The selected provider rejected all bounded attempts due to quota."""


class ProviderAuthenticationError(ModelProviderError):
    """The selected provider rejected the supplied credentials."""


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


def _status_code(exc: Exception) -> int | None:
    for candidate in (getattr(exc, "status_code", None), getattr(exc, "status", None),
                      getattr(getattr(exc, "response", None), "status_code", None),
                      getattr(exc, "code", None)):
        if isinstance(candidate, int):
            return candidate
    match = re.search(r"(?:status|http)[^0-9]*(\d{3})", str(exc), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _provider_failure(provider: str, exc: Exception) -> ModelProviderError:
    status = _status_code(exc)
    if status == 429:
        return ProviderRateLimitError(
            f"{provider} rate limit or quota was exceeded after 3 attempts. "
            "Wait for the quota window to reset, check billing/quota, or use another provider.")
    if status in {401, 403}:
        return ProviderAuthenticationError(
            f"{provider} rejected the API key. Check that the key is valid and permitted to use the configured model.")
    return ModelProviderError(f"{provider} request failed. Check the provider status and model configuration.")


def _with_retries(provider: str, operation, sleep_fn=time.sleep):
    for attempt in range(3):
        try:
            return operation()
        except Exception as exc:
            if _status_code(exc) == 429 and attempt < 2:
                LOGGER.warning("provider_rate_limit_retry", extra={"provider": provider, "attempt": attempt + 1})
                sleep_fn(2 ** attempt)
                continue
            raise _provider_failure(provider, exc) from None


class MistralLLMProvider:
    """Explicit opt-in cloud fallback, never instantiated on the default path."""

    def __init__(self, api_key: str | None = None, client=None, sleep_fn=time.sleep):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is not configured")
        if client is None:
            from mistralai import Mistral
            client = Mistral(api_key=key)
        self.client = client
        self.sleep_fn = sleep_fn
        self.model = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = _with_retries("Mistral", lambda: self.client.chat.complete(
            model=self.model, messages=[{"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_prompt}]), self.sleep_fn)
        return response.choices[0].message.content

    def generate_structured(self, system_prompt: str, user_prompt: str,
                            response_model: Any = None) -> Any:
        return _parse_json(self.generate(system_prompt, user_prompt), response_model)


class GeminiLLMProvider:
    """Gemini REST adapter with native JSON-schema output and no extra SDK dependency."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout: float = 120, sleep_fn=time.sleep):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = timeout
        self.sleep_fn = sleep_fn

    def _request(self, system_prompt: str, user_prompt: str,
                 response_model: Any = None) -> str:
        generation_config: dict[str, Any] = {"temperature": 0}
        if response_model is not None:
            generation_config.update({
                "responseMimeType": "application/json",
                "responseJsonSchema": TypeAdapter(response_model).json_schema(),
            })
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }
        request = Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )

        def send():
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)

        result = _with_retries("Gemini", send, self.sleep_fn)
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelProviderError("Gemini returned no usable response content.") from exc

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._request(system_prompt, user_prompt)

    def generate_structured(self, system_prompt: str, user_prompt: str,
                            response_model: Any = None) -> Any:
        return _parse_json(self._request(system_prompt, user_prompt, response_model), response_model)


class LLMClient:
    """Compatibility facade for existing agents; local inference is the default."""

    def __init__(self, provider: LLMProvider | None = None,
                 cloud_provider: LLMProvider | None = None):
        selection = os.getenv("LLM_PROVIDER", "local").casefold()
        if selection not in {"local", "mistral", "gemini"}:
            raise ValueError("LLM_PROVIDER must be 'local', 'mistral', or 'gemini'")
        if selection in {"mistral", "gemini"} and os.getenv("ALLOW_CLOUD_FALLBACK", "false").casefold() != "true":
            raise ValueError("Cloud model use requires ALLOW_CLOUD_FALLBACK=true")
        cloud = GeminiLLMProvider if selection == "gemini" else MistralLLMProvider
        self.provider = provider or (LocalLLMProvider() if selection == "local" else cloud())
        self.cloud_provider = cloud_provider
        self.allow_cloud_fallback = os.getenv("ALLOW_CLOUD_FALLBACK", "false").casefold() == "true"

    def _call(self, method: str, *args):
        try:
            return getattr(self.provider, method)(*args)
        except (LocalModelUnavailable, StructuredOutputError, ModelProviderError):
            if not self.allow_cloud_fallback or isinstance(self.provider, (MistralLLMProvider, GeminiLLMProvider)):
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
