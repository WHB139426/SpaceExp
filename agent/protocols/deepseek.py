"""DeepSeek V4.1 context options and DSML tool-call parsing."""

from __future__ import annotations

import re
import uuid

from ..types import Message, ModelInput, ModelOutput

from .common import parse_value, prepare_messages


TOOL_CALLS_RE = re.compile(
    r"<｜DSML｜ calls>\s*(.*?)\s*</｜DSML｜ calls>", re.S
)
TOOL_CALL_RE = re.compile(
    r'<｜DSML｜ invoke name="([^"]+)">\s*(.*?)\s*</｜DSML｜ invoke>', re.S
)
PARAMETER_RE = re.compile(
    r'<｜DSML｜ parameter name="([^"]+)" string="(true|false)">'
    r"(.*?)</｜DSML｜ parameter>",
    re.S,
)
STOP_TOKEN_RE = re.compile(r"(?:<｜end▁of▁sentence｜>\s*)+$")


class DeepSeekProtocol:
    def __init__(self, *, reasoning_effort: int = 75) -> None:
        if type(reasoning_effort) is not int or not 1 <= reasoning_effort <= 100:
            raise ValueError("reasoning_effort must be between 1 and 100")
        self.reasoning_effort = reasoning_effort

    def render(
        self, messages: list[Message], tools: list[dict[str, object]]
    ) -> ModelInput:
        return ModelInput(
            messages=prepare_messages(messages),
            tools=tools,
            chat_template_kwargs={
                "thinking_mode": "thinking",
                "reasoning_effort": self.reasoning_effort,
            },
        )

    def parse(self, text: ModelOutput) -> Message:
        if not isinstance(text, str):
            raise TypeError("Native vLLM protocol expects tool-call text")
        text = STOP_TOKEN_RE.sub("", text).strip()
        if "</think>" not in text:
            raise RuntimeError("DeepSeek response is missing the closing </think> tag")

        reasoning, _, content = text.partition("</think>")
        reasoning = reasoning.removeprefix("<think>").strip()
        tool_calls = []

        for block in TOOL_CALLS_RE.finditer(content):
            for call in TOOL_CALL_RE.finditer(block.group(1)):
                arguments = {}
                for parameter in PARAMETER_RE.finditer(call.group(2)):
                    name, is_string, value = parameter.groups()
                    arguments[name] = (
                        value if is_string == "true" else parse_value(value)
                    )
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": call.group(1),
                            "arguments": arguments,
                        },
                    }
                )

        message: Message = {
            "role": "assistant",
            "content": TOOL_CALLS_RE.sub("", content).strip(),
        }
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message
