import os

from google import genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from eval.adapters.base import BaseModelAdapter, GenerationResult

# gemini-3-flash pricing per 1M tokens (estimated)
INPUT_PRICE = 0.15
OUTPUT_PRICE = 0.60


class GeminiAdapter(BaseModelAdapter):
    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=60),
        retry=retry_if_exception_type((ResourceExhausted, GoogleAPIError))
    )
    def generate(self, prompt: str, *, system_prompt: str | None = None) -> GenerationResult:
        config = genai.types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=system_prompt,
        )

        response = self._client.models.generate_content(
            model=self._model,
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
