"""OpenAI adapter for gpt-5."""

import os

from openai import OpenAI, RateLimitError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


from eval.adapters.base import BaseModelAdapter, GenerationResult

# gpt-5 pricing per 1M tokens (estimated)
INPUT_PRICE = 5.00
OUTPUT_PRICE = 15.00


class OpenAIAdapter(BaseModelAdapter):
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o") 

    @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=60),
            retry=retry_if_exception_type((RateLimitError, APIError))
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

        if not response.choices:
            return GenerationResult(text="", input_tokens=0, output_tokens=0)
        return GenerationResult(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens or 0 if usage else 0,
            output_tokens=usage.completion_tokens or 0 if usage else 0,
        )

    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * INPUT_PRICE + (output_tokens / 1_000_000) * OUTPUT_PRICE
