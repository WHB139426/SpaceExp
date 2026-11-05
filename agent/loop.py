"""Model calls, tool execution and conversation history."""

from __future__ import annotations

import json
from pathlib import Path

from .types import ChatProtocol, Message, Model, Tool, ToolOutput


def trim_images(messages: list[Message], max_images: int) -> None:
    """Replace older images as whole message groups, retaining their text."""
    if max_images == -1:
        return
    remaining = max_images
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count = sum(part.get("type") in {"image", "image_url"} for part in content)
        if count <= remaining:
            remaining -= count
        else:
            remaining = 0
            for index, part in enumerate(content):
                if part.get("type") not in {"image", "image_url"}:
                    continue
                content[index] = {"type": "text", "text": "[older image removed from conversation]"}


class Agent:
    def __init__(
        self,
        model: Model,
        protocol: ChatProtocol,
        tools: list[Tool],
        *,
        system_prompt: str,
        max_steps: int,
        max_conversation_images: int,
        conversation_path: str | Path | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_conversation_images != -1 and max_conversation_images < 1:
            raise ValueError("max_conversation_images must be -1 or at least 1")
        self.model = model
        self.protocol = protocol
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("tool names must be unique")
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_conversation_images = max_conversation_images
        self.conversation_path = Path(conversation_path) if conversation_path is not None else None
        self.reset()

    def reset(self) -> None:
        self.messages: list[Message] = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self._save()

    def _save(self) -> None:
        if self.conversation_path is None:
            return
        self.conversation_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.conversation_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.messages, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.conversation_path)

    def run(self, prompt: str | list[dict[str, object]]) -> str:
        self.messages.append({"role": "user", "content": prompt})
        schemas = [tool.schema() for tool in self.tools.values()]
        for _ in range(self.max_steps):
            trim_images(self.messages, self.max_conversation_images)
            self._save()
            assistant = self.protocol.parse(self.model.generate(self.protocol.render(self.messages, schemas)))
            self.messages.append(assistant)
            self._save()
            calls = assistant.get("tool_calls", [])
            if not calls:
                return str(assistant.get("content", ""))

            results = []
            attachments = []
            for call in calls:
                result, extra_messages = self._execute_tool(call)
                results.append(result)
                attachments.extend(extra_messages)
            # Complete every tool call before adding visual observation messages.
            self.messages.extend(results)
            self.messages.extend(attachments)
            self._save()
        raise RuntimeError(f"agent exceeded max_steps={self.max_steps}")

    def _execute_tool(self, call: Message) -> tuple[Message, tuple[Message, ...]]:
        function = call["function"]
        name = function["name"]
        attachments = ()
        try:
            if name not in self.tools:
                raise ValueError(f"unknown tool: {name}")
            value = self.tools[name].function(**function["arguments"])
            if isinstance(value, ToolOutput):
                attachments = value.messages
                value = value.value
            payload = {"ok": True, "result": value}
        except Exception as exc:
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": name,
            "content": json.dumps(payload, ensure_ascii=False),
        }, attachments
