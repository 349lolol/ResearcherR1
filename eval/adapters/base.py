from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int


class BaseModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, system_prompt: str | None = None) -> GenerationResult:
        pass

    @abstractmethod
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        pass
