"""OpenAI-compatible embedding provider for API-based embeddings."""

from __future__ import annotations

import json
import os
import urllib.request


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        endpoint: str,
        model: str,
        dimensions: int,
        api_key_env: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._dims = dimensions
        self._api_key_env = api_key_env
        self._timeout_seconds = timeout_seconds

    def model_id(self) -> str:
        return self._model

    def dimensions(self) -> int:
        return self._dims

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self._model, "input": texts}
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key_env:
            key = os.environ.get(self._api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"

        req = urllib.request.Request(
            f"{self._endpoint}/v1/embeddings",
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))

        vectors = [item["embedding"] for item in body.get("data", [])]
        if len(vectors) != len(texts):
            raise ValueError("Embedding API returned unexpected number of vectors")
        return vectors
