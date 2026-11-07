"""vLLM Chat Completions and OpenAI Responses requests."""

from __future__ import annotations

import base64
import copy
import mimetypes
from pathlib import Path

from .types import ModelInput


def image_url(url: str) -> str:
    """Embed a local image only for transport; history keeps its file path."""
    if url.startswith(('data:', 'http://', 'https://')):
        return url
    path = Path(url).expanduser()
    media_type = mimetypes.guess_type(path.name)[0]
    if media_type is None or not media_type.startswith('image/'):
        raise ValueError(f'Cannot determine image type: {path}')
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:{media_type};base64,{encoded}'


class VLLMBackend:
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        timeout: float = 180,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=0)

    def generate(self, model_input: ModelInput) -> str:
        # History retains file paths. Only the outgoing request embeds image bytes.
        messages = copy.deepcopy(model_input.messages)
        for message in messages:
            content = message.get('content')
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get('type') != 'image_url':
                    continue
                part['image_url']['url'] = image_url(part['image_url']['url'])
        extra_body = {
            "top_k": self.top_k,
            "skip_special_tokens": False,
            "chat_template_kwargs": model_input.chat_template_kwargs,
        }
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": extra_body,
        }
        if model_input.tools:
            request["tools"] = model_input.tools
            request["tool_choice"] = "none"
        response = self.client.chat.completions.create(**request)
        if not response.choices:
            raise RuntimeError("vLLM returned no choices")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(f"vLLM generation reached max_tokens={self.max_tokens} before completion")
        message = choice.message
        if getattr(message, "tool_calls", None):
            raise RuntimeError("vLLM returned parsed tool calls; disable its automatic tool parser")
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning", None)
        if reasoning is not None:
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        return content


class OpenAIBackend:
    def __init__(
        self, model: str, *, max_output_tokens: int, reasoning_effort: str = 'high',
        reasoning_summary: str = 'auto', base_url: str = 'https://api.openai.com/v1',
        api_key: str | None = None, timeout: float = 300,
    ) -> None:
        from openai import OpenAI

        if reasoning_effort not in {'low', 'medium', 'high', 'xhigh', 'max'}:
            raise ValueError('Unsupported GPT-6 reasoning effort')
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning = {'effort': reasoning_effort, 'summary': reasoning_summary}
        # None lets the SDK read OPENAI_API_KEY. Never reuse the local vLLM URL.
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def generate(self, model_input: ModelInput) -> dict[str, object]:
        items = copy.deepcopy(model_input.messages)
        for item in items:
            content = item.get('content')
            if isinstance(content, list):
                for part in content:
                    if part.get('type') == 'input_image':
                        part['image_url'] = image_url(part['image_url'])
        response = self.client.responses.create(
            model=self.model, instructions=model_input.instructions, input=items,
            tools=model_input.tools, tool_choice='auto', parallel_tool_calls=False,
            reasoning=self.reasoning, max_output_tokens=self.max_output_tokens,
            store=False, include=['reasoning.encrypted_content'], truncation='disabled',
        )
        if response.status != 'completed':
            raise RuntimeError(
                f'OpenAI response {response.id} did not complete: '
                f'{response.status}; {response.incomplete_details}; {response.error}'
            )
        return response.model_dump(mode='json', exclude_none=True)
