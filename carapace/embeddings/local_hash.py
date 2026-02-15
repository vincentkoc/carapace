"""Fast local deterministic hash embedding provider."""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class LocalHashEmbeddingProvider:
    def __init__(self, dims: int = 256, model: str = "hash-embed-v1") -> None:
        if dims <= 0:
            raise ValueError("dims must be positive")
        self._dims = dims
        self._model = model

    def model_id(self) -> str:
        return self._model

    def dimensions(self) -> int:
        return self._dims

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        vector = [0.0] * self._dims
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self._dims
            sign = -1.0 if int(digest[-1], 16) % 2 else 1.0
            vector[idx] += sign

        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
