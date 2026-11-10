"""Construct a backend and protocol from one explicit model registry."""

from .backend import OpenAIBackend, VLLMBackend
from .protocols import MODEL_ADAPTERS, create_protocol


BACKENDS = {
    'vllm': (VLLMBackend, ('max_tokens', 'temperature', 'top_p', 'top_k')),
    'openai': (OpenAIBackend, ('max_output_tokens', 'reasoning_effort', 'reasoning_summary')),
}
PROTOCOL_OPTIONS = {
    'qwen': ('enable_thinking', 'reasoning_effort', 'preserve_thinking'),
    'glm': ('reasoning_effort', 'preserve_thinking'),
    'deepseek': ('reasoning_effort',),
    'openai': (),
}


def create_model(model, model_config):
    if model not in MODEL_ADAPTERS:
        raise ValueError(f'No adapter registered for {model!r}; add it to MODEL_ADAPTERS')
    adapter = MODEL_ADAPTERS[model]
    protocol = create_protocol(model, **{
        key: model_config[key] for key in PROTOCOL_OPTIONS[adapter['protocol']]
        if key in model_config
    })
    backend_class, keys = BACKENDS[adapter['backend']]
    backend = backend_class(model, **model_config['server_config'], **{
        key: model_config[key] for key in keys
    })
    return backend, protocol


def model_config_snapshot(model_config):
    """Keep settings in run metadata without recording an explicit API key."""
    return {**model_config, 'server_config': {
        key: value for key, value in model_config['server_config'].items() if key != 'api_key'
    }}
