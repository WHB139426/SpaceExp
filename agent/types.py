"""Shared data and structural interfaces for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


Message = dict[str, object]
ModelOutput = str | dict[str, object]


@dataclass(frozen=True)
class ModelInput:
    messages: list[Message]
    tools: list[dict[str, object]]
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    instructions: str = ""


class Model(Protocol):
    def generate(self, model_input: ModelInput) -> ModelOutput: ...


class ChatProtocol(Protocol):
    def render(self, messages: list[Message], tools: list[dict[str, object]]) -> ModelInput: ...

    def parse(self, output: ModelOutput) -> Message: ...


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, object]
    function: Callable[..., object]

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolOutput:
    """Tool result and optional messages, including visual observations."""

    value: object
    messages: tuple[Message, ...] = ()
