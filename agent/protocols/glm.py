"""GLM context rendering and tool-call parsing."""

from __future__ import annotations

import json
import re
import uuid

from ..types import Message, ModelInput, ModelOutput

from .common import prepare_messages


TOOL_CALL_RE = re.compile(r"<tool_call>\s*([^<\s]+)\s*(.*?)\s*</tool_call>", re.S)
ARGUMENT_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.S
)
STOP_TOKEN_RE = re.compile(
    r"(?:(?:<\|endoftext\|>|<\|user\|>|<\|observation\|>)\s*)+$"
)


class GLMProtocol:
    def __init__(
        self,
        *,
        reasoning_effort: str = "max",
        preserve_thinking: bool = True,
    ) -> None:
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be 'low', 'high', or 'max'")
        self.reasoning_effort = reasoning_effort
        self.preserve_thinking = preserve_thinking

    def render(
        self, messages: list[Message], tools: list[dict[str, object]]
    ) -> ModelInput:
        return ModelInput(
            messages=prepare_messages(messages),
            tools=tools,
            chat_template_kwargs={
                "reasoning_effort": self.reasoning_effort,
                "clear_thinking": not self.preserve_thinking,
            },
        )

    def parse(self, text: ModelOutput) -> Message:
        if not isinstance(text, str):
            raise TypeError("Native vLLM protocol expects tool-call text")
        text = STOP_TOKEN_RE.sub("", text).strip()
        if "</think>" in text:
            reasoning, _, content = text.partition("</think>")
            reasoning = reasoning.removeprefix("<think>").strip()
            content = content.strip()
        else:
            reasoning, content = "", text

        tool_calls = []
        for match in TOOL_CALL_RE.finditer(content):
            arguments = {}
            for item in ARGUMENT_RE.finditer(match.group(2)):
                value = item.group(2).strip()
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
                arguments[item.group(1).strip()] = value
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": match.group(1).strip(),
                        "arguments": arguments,
                    },
                }
            )

        message: Message = {
            "role": "assistant",
            "content": TOOL_CALL_RE.sub("", content).strip(),
        }
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message
