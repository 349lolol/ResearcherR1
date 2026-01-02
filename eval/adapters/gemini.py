"""Gemini adapter for gemini-3-flash."""

import os

from google import genai

from eval.adapters.base import BaseModelAdapter, GenerationResult

# gemini-3-flash pricing per 1M tokens (estimated)
INPUT_PRICE = 0.15
OUTPUT_PRICE = 0.60


class GeminiAdapter(BaseModelAdapter):
    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> GenerationResult:
        config = genai.types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=system_prompt,
        )

        response = self._client.models.generate_content(
            model="gemini-3-flash",
            contents=prompt,
            config=config,
        )

        usage = response.usage_metadata
        return GenerationResult(
            text=response.text or "",
            input_tokens=usage.prompt_token_count or 0 if usage else 0,
            output_tokens=usage.candidates_token_count or 0 if usage else 0,
        )

    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * INPUT_PRICE + (output_tokens / 1_000_000) * OUTPUT_PRICE
