"""Model-specific message preparation and output parsing."""

from ..types import ChatProtocol
from .deepseek import DeepSeekProtocol
from .glm import GLMProtocol
from .openai import OpenAIProtocol
from .qwen import QwenProtocol


PROTOCOLS = {"qwen": QwenProtocol, "glm": GLMProtocol,
             "deepseek": DeepSeekProtocol, "openai": OpenAIProtocol}
MODEL_ADAPTERS = {
    "glm-5.3-flash": {"backend": "vllm", "protocol": "glm"},
    "deepseek-v4.1-flash": {"backend": "vllm", "protocol": "deepseek"},
    "qwen3.5-9b": {"backend": "vllm", "protocol": "qwen"},
    "qwen3.8-27b": {"backend": "vllm", "protocol": "qwen"},
    "qwen3.8-flash-next": {"backend": "vllm", "protocol": "qwen"},
    "gpt-6-astra": {"backend": "openai", "protocol": "openai"},
    "gpt-6.1-sol": {"backend": "openai", "protocol": "openai"},
    "gpt-6.1-terra": {"backend": "openai", "protocol": "openai"},
    "gpt-6-sol": {"backend": "openai", "protocol": "openai"},
    "gpt-6-luna": {"backend": "openai", "protocol": "openai"},
}
MODEL_PROTOCOLS = {model: adapter['protocol'] for model, adapter in MODEL_ADAPTERS.items()}


def create_protocol(model: str, **options) -> ChatProtocol:
    if model not in MODEL_PROTOCOLS:
        raise ValueError(f"No protocol registered for model: {model!r}. Add it to MODEL_ADAPTERS.")
    name = MODEL_PROTOCOLS[model]
    if name not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {name!r}. Available: {', '.join(PROTOCOLS)}")
    return PROTOCOLS[name](**options)
