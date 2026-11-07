"""Construct a backend and protocol from one explicit model registry."""

from .backend import OpenAIBackend, VLLMBackend
from .protocols import MODEL_ADAPTERS, create_protocol


BACKENDS = {
    'vllm': (VLLMBackend, ('max_tokens', 'temperature', 'top_p', 'top_k')),
    'openai': (OpenAIBackend, ('max_output_tokens', 'reasoning_effort', 'reasoning_summary')),
}
PROTOCOL_OPTIONS = {
    'qwen': ('enable_thinking', 'reasoning_effort', 'preserve_thinking'),
    'openai': (),
}


def create_model(model, generation_config, server_configs):
    if model not in MODEL_ADAPTERS:
        raise ValueError(f'No adapter registered for {model!r}; add it to MODEL_ADAPTERS')
    adapter = MODEL_ADAPTERS[model]
    protocol = create_protocol(model, **{
        key: generation_config[key] for key in PROTOCOL_OPTIONS[adapter['protocol']]
    })
    backend_class, keys = BACKENDS[adapter['backend']]
    backend = backend_class(model, **server_configs[adapter['backend']], **{
        key: generation_config[key] for key in keys
    })
    return backend, protocol
