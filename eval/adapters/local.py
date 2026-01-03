"""Local LLM adapter for LM Studio (OpenAI-compatible API)."""

import os

from openai import OpenAI

from eval.adapters.base import BaseModelAdapter, GenerationResult


class LocalAdapter(BaseModelAdapter):
    def __init__(self, model: str = "ministral-8b-instruct"):
        self._model = model
        self._client = OpenAI(
            api_key="lm-studio",  # LM Studio doesn't need a real key
            base_url=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
        )

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
        )

        usage = response.usage
        return GenerationResult(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens or 0 if usage else 0,
            output_tokens=usage.completion_tokens or 0 if usage else 0,
        )

    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # Local = free
