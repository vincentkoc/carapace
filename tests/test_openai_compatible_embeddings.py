import json

import pytest

from carapace.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_openai_provider_model_and_dimensions() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        endpoint="https://example.test",
        model="text-embedding-3-small",
        dimensions=256,
    )
    assert provider.model_id() == "text-embedding-3-small"
    assert provider.dimensions() == 256


def test_openai_provider_embed_texts_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _fake_urlopen(request, timeout: float):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = request.data.decode("utf-8")
        return _FakeResponse({"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setenv("TEST_API_KEY", "secret-token")

    provider = OpenAICompatibleEmbeddingProvider(
        endpoint="https://example.test/",
        model="text-embedding-3-small",
        dimensions=256,
        api_key_env="TEST_API_KEY",
        timeout_seconds=3.5,
    )
    vectors = provider.embed_texts(["one", "two"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "https://example.test/v1/embeddings"
    assert captured["timeout"] == 3.5
    body = json.loads(captured["body"])
    assert body == {"model": "text-embedding-3-small", "input": ["one", "two"]}
    assert captured["headers"]["Authorization"] == "Bearer secret-token"


def test_openai_provider_embed_texts_validates_vector_count(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request, timeout: float):  # noqa: ANN001
        return _FakeResponse({"data": [{"embedding": [0.1, 0.2]}]})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    provider = OpenAICompatibleEmbeddingProvider(
        endpoint="https://example.test",
        model="text-embedding-3-small",
        dimensions=256,
    )

    with pytest.raises(ValueError, match="unexpected number of vectors"):
        provider.embed_texts(["one", "two"])
