from __future__ import annotations

from typing import Iterator, List, Optional


class OllamaLLM:
    """Thin wrapper over the Ollama Python client for qwen2.5 models."""

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host)
        return self._client

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        client = self._get_client()
        options = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat(model=self.model, messages=messages, options=options)
        return resp["message"]["content"]

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        client = self._get_client()
        options = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        for chunk in client.chat(
            model=self.model, messages=messages, options=options, stream=True
        ):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
