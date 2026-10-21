from abc import ABC, abstractmethod
import logging
from time import perf_counter
from typing import Any

from pydantic import BaseModel, TypeAdapter

from core.logging import state_summary


class AgentContractError(ValueError):
    """Raised when an agent's declared input or output contract is not met."""

    def __init__(self, agent: str, field: str, direction: str):
        self.agent = agent
        self.field = field
        self.direction = direction
        super().__init__(f"{agent} requires {direction} field '{field}'")


class BaseAgent(ABC):
    name = "base_agent"
    input_fields: tuple[str, ...] = ()
    output_fields: tuple[str, ...] = ()

    def __init__(self, llm=None):
        self._llm = llm
        self.logger = logging.getLogger(f"mortgage.agent.{self.name}")

    @property
    def llm(self):
        """Create the configured LLM only when an agent actually needs it."""
        if self._llm is None:
            from core.llm import LLMClient
            self._llm = LLMClient()
        return self._llm

    def validate_input(self, state) -> None:
        if state is None:
            raise ValueError("Mortgage state is required")
        for field in self.input_fields:
            if getattr(state, field, None) is None:
                raise AgentContractError(self.name, field, "input")

    def validate_output(self, result: Any) -> None:
        if result is None:
            raise ValueError(f"{self.name} produced no result")
        for field in self.output_fields:
            if getattr(result, field, None) is None:
                raise AgentContractError(self.name, field, "output")

    def update_state(self, state, result):
        """Compatibility hook: existing agents return their mutated state."""
        return result

    def chat_json(self, system_prompt: str, user_prompt: str, response_model=None):
        """Typed local output when available; preserve two-argument test doubles."""
        from core.llm import LLMClient
        if isinstance(self.llm, LLMClient):
            result = self.llm.chat_json(system_prompt, user_prompt, response_model)
        else:
            result = self.llm.chat_json(system_prompt, user_prompt)
            if response_model is not None:
                result = TypeAdapter(response_model).validate_python(result)
        if isinstance(result, BaseModel):
            return result.model_dump()
        if isinstance(result, list):
            return [item.model_dump() if isinstance(item, BaseModel) else item for item in result]
        return result

    def execute(self, state, *args, **kwargs):
        """Standard lifecycle used by workflow nodes and test doubles."""
        started = perf_counter()
        application_id = getattr(state, "application_id", None)
        self.logger.info("agent_started", extra={"agent": self.name, "application_id": application_id})
        try:
            self.validate_input(state)
            result = self.run(state, *args, **kwargs)
            result = self.update_state(state, result)
            self.validate_output(result)
        except Exception:
            self.logger.exception("agent_failed", extra={
                "agent": self.name, "application_id": application_id,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            })
            raise
        self.logger.info("agent_completed", extra={
            "agent": self.name, "application_id": application_id,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            **state_summary(result),
        })
        return result

    @abstractmethod
    def run(self, state):
        pass
