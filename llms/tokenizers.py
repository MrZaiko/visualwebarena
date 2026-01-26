from typing import Any

import tiktoken
from transformers import LlamaTokenizer  # type: ignore


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            if model_name == "gpt-5.1-2025-11-13":
                self.tokenizer = tiktoken.get_encoding("o200k_base")
            elif model_name == "claude-sonnet-4-5-20250929":
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            elif model_name == "openai/gpt-oss-120b":
                self.tokenizer = tiktoken.get_encoding("o200k_harmony")
            else:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            # self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        elif provider == "anthropic":
            # Use cl100k_base as a reasonable approximation for Claude tokenization
            # Claude's actual tokenizer is not publicly available, but cl100k_base
            # provides a reasonable approximation for token counting purposes
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
