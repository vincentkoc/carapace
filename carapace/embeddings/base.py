"""Embedding provider abstractions."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    def model_id(self) -> str:
        ...

    def dimensions(self) -> int:
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...
