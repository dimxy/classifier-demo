from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv

load_dotenv()


class OpenAILLM:
    """Thin wrapper over the OpenAI Python client.

    Each node can point at its own OpenAI-compatible endpoint via ``base_url``.
    The API key is read at call time from the environment variable named in
    ``api_key_env`` (default ``OPENAI_API_KEY``), with values from a local
    ``.env`` file merged in via python-dotenv, so keys never live in config files.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        if not base_url:
            raise ValueError(
                f"OpenAILLM for model {model!r} requires a non-empty base_url"
            )
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get(self.api_key_env, "")
            self._client = OpenAI(
                api_key=api_key or "missing",
                base_url=self.base_url,
            )
        return self._client

    def _build_kwargs(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: Optional[int],
        stop: Optional[List[str]],
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        return kwargs

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        client = self._get_client()
        kwargs = self._build_kwargs(prompt, system, max_tokens, stop)
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        client = self._get_client()
        kwargs = self._build_kwargs(prompt, system, max_tokens, stop)
        kwargs["stream"] = True
        for chunk in client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content or ""
            if piece:
                yield piece
