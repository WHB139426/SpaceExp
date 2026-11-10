"""Shared helpers for native vLLM tool-call protocols."""

import copy
import json

from ..types import Message


def parse_value(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def prepare_messages(messages: list[Message]) -> list[Message]:
    prepared = copy.deepcopy(messages)
    for message in prepared:
        for call in message.get('tool_calls', []):
            function = call['function']
            if not isinstance(function['arguments'], str):
                function['arguments'] = json.dumps(function['arguments'], ensure_ascii=False)
    return prepared
