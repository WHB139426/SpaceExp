"""Qwen thinking messages and XML tool-call parsing."""

from __future__ import annotations

import copy
import json
import re
import uuid

from ..types import Message, ModelInput


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*</tool_call>", re.S,
)
PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.S)
STOP_TOKEN_RE = re.compile(r"(?:<\|im_end\|>\s*)+$")


def _parse_value(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class QwenProtocol:
    """Qwen chat-template options and text tool-call protocol."""

    def __init__(self, *, enable_thinking: bool, reasoning_effort: str, preserve_thinking: bool) -> None:
        if reasoning_effort not in {"low", "medium", "xhigh"}:
            raise ValueError("reasoning_effort must be 'low', 'medium', or 'xhigh'")
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.preserve_thinking = preserve_thinking

    def render(self, messages: list[Message], tools: list[dict[str, object]]) -> ModelInput:
        prepared = copy.deepcopy(messages)
        for message in prepared:
            for call in message.get("tool_calls", []):
                function = call["function"]
                function["arguments"] = json.dumps(function["arguments"], ensure_ascii=False)
        return ModelInput(
            messages=prepared,
            tools=tools,
            chat_template_kwargs={
                "enable_thinking": self.enable_thinking,
                "reasoning_effort": self.reasoning_effort,
                "preserve_thinking": self.preserve_thinking,
            },
        )

    def parse(self, text: str) -> Message:
        text = STOP_TOKEN_RE.sub("", text).strip()
        reasoning = ""
        content = text
        if "</think>" in text:
            reasoning, _, content = text.partition("</think>")
            reasoning = reasoning.removeprefix("<think>").strip()
        elif self.enable_thinking and not text:
            raise RuntimeError("vLLM returned an empty response in thinking mode")
        calls = []
        for match in TOOL_CALL_RE.finditer(content):
            arguments = {
                parameter.group(1).strip(): _parse_value(parameter.group(2).strip())
                for parameter in PARAMETER_RE.finditer(match.group(2))
            }
            calls.append({
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": match.group(1).strip(), "arguments": arguments},
            })
        message: Message = {"role": "assistant", "content": TOOL_CALL_RE.sub("", content).strip()}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        return message
