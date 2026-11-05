"""vLLM chat requests with native tool-call text returned to the protocol."""

from __future__ import annotations

import base64
import copy
import mimetypes
from pathlib import Path

from .types import ModelInput


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
                url = part['image_url']['url']
                if url.startswith(('data:', 'http://', 'https://')):
                    continue
                path = Path(url).expanduser()
                media_type = mimetypes.guess_type(path.name)[0]
                if media_type is None or not media_type.startswith('image/'):
                    raise ValueError(f'Cannot determine image type: {path}')
                encoded = base64.b64encode(path.read_bytes()).decode('ascii')
                part['image_url']['url'] = f'data:{media_type};base64,{encoded}'
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
