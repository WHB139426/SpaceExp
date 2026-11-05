"""Model-specific message preparation and output parsing."""

from ..types import ChatProtocol
from .qwen import QwenProtocol


PROTOCOLS = {"qwen": QwenProtocol}
MODEL_PROTOCOLS = {
    "qwen3.8-27b": "qwen",
    "qwen3.8-flash-next": "qwen",
}


def create_protocol(model: str, **options) -> ChatProtocol:
    if model not in MODEL_PROTOCOLS:
        raise ValueError(f"No protocol registered for model: {model!r}. Add it to MODEL_PROTOCOLS.")
    name = MODEL_PROTOCOLS[model]
    if name not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {name!r}. Available: {', '.join(PROTOCOLS)}")
    return PROTOCOLS[name](**options)
