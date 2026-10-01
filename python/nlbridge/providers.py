from __future__ import annotations
import http.client
import importlib.metadata
import os
import re
import threading
from urllib.parse import urlsplit
from .common import BridgeError, ModelError, canonical, strict_json


class JsonHTTP:
    """Persistent connections per worker, bounded responses, no extra HTTP package."""

    def __init__(self, base_url, timeout=45, api_key_env=""):
        p = urlsplit(base_url)
        if (
            p.scheme not in {"http", "https"}
            or not p.hostname
            or p.username
            or p.password
            or p.query
            or p.fragment
        ):
            raise BridgeError("invalid model base URL")
        self.p = p
        self.timeout = timeout
        self.api_key_env = api_key_env
        self.local = threading.local()

    def post(self, suffix, body):
        raw = canonical(body).encode()
        if len(raw) > 4 * 1024 * 1024:
            raise ModelError("model request exceeds 4 MiB")
        if not getattr(self.local, "conn", None):
            cls = (
                http.client.HTTPSConnection
                if self.p.scheme == "https"
                else http.client.HTTPConnection
            )
            self.local.conn = cls(self.p.hostname, self.p.port, timeout=self.timeout)
        conn = self.local.conn
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ModelError("configured API key environment variable is missing")
            headers["Authorization"] = "Bearer " + key
        try:
            conn.request(
                "POST",
                self.p.path.rstrip("/") + "/" + suffix.lstrip("/"),
                body=raw,
                headers=headers,
            )
            response = conn.getresponse()
            payload = response.read(4 * 1024 * 1024 + 1)
            if response.status != 200:
                raise ModelError(f"model endpoint HTTP {response.status}")
            if len(payload) > 4 * 1024 * 1024:
                raise ModelError("model response exceeds 4 MiB")
            result = strict_json(payload)
            if not isinstance(result, dict):
                raise ModelError("model endpoint returned a non-object envelope")
            return result
        except (OSError, http.client.HTTPException, BridgeError) as e:
            conn.close()
            self.local.conn = None
            if isinstance(e, ModelError):
                raise
            raise ModelError("model transport or JSON response failed") from e


class ChatModel:
    def __init__(
        self,
        provider,
        model,
        base_url,
        revision,
        timeout=45,
        max_tokens=768,
        temperature=0.0,
        api_key_env="",
    ):
        if provider not in {"ollama", "openai"}:
            raise BridgeError("model provider must be ollama or openai")
        if not model or not revision:
            raise BridgeError("model and deployment revision are required")
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.http = JsonHTTP(base_url, timeout, api_key_env)
        self.descriptor = {
            "provider": provider,
            "model": model,
            "revision": revision,
            "base_url": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def generate(self, messages, schema):
        if self.provider == "ollama":
            data = self.http.post(
                "api/chat",
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "format": schema,
                    "think": False,
                    "keep_alive": -1,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            )
            if data.get("done_reason") in {"length", "max_tokens"} or not data.get(
                "done", True
            ):
                raise ModelError("model output was truncated")
            text = data.get("message", {}).get("content")
        else:
            data = self.http.post(
                "chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "nlbridge",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                },
            )
            choices = data.get("choices", [])
            if not choices or choices[0].get("finish_reason") not in {"stop"}:
                raise ModelError("model did not finish a structured response")
            text = choices[0].get("message", {}).get("content")
        if not isinstance(text, str):
            raise ModelError("model returned no text content")
        # Syntax errors can be repaired by Runtime with explicit feedback.
        return strict_json(text)


class SentenceEmbedder:
    def __init__(
        self,
        model,
        revision,
        backend="onnx",
        query_prefix="",
        document_prefix="",
        local_files_only=False,
    ):
        if not isinstance(revision, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", revision
        ):
            raise BridgeError("embedding model requires an immutable revision")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise BridgeError("install nlbridge[models] to use local embeddings") from e
        self.model = SentenceTransformer(
            model, revision=revision, backend=backend, local_files_only=local_files_only
        )
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.lock = threading.Lock()
        self.descriptor = {
            "provider": "sentence_transformers",
            "libraries": {
                p: importlib.metadata.version(p)
                for p in ("sentence-transformers", "transformers", "tokenizers")
            },
            "model": model,
            "revision": revision,
            "backend": backend,
            "dim": self.dim,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
        }

    def embed_documents(self, texts):
        with self.lock:
            return self.model.encode(
                [self.document_prefix + t for t in texts],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

    def embed_query(self, text):
        with self.lock:
            return self.model.encode(
                [self.query_prefix + text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].tolist()


class HTTPEmbedder:
    def __init__(
        self,
        provider,
        model,
        revision,
        base_url,
        dim,
        query_prefix="",
        document_prefix="",
        timeout=45,
        api_key_env="",
    ):
        if provider not in {"ollama", "openai"} or not revision or dim <= 0:
            raise BridgeError("embedding provider, revision, and dimension required")
        self.provider = provider
        self.model = model
        self.dim = dim
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.http = JsonHTTP(base_url, timeout, api_key_env)
        self.descriptor = {
            "provider": provider,
            "model": model,
            "revision": revision,
            "base_url": base_url,
            "dim": dim,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
        }

    def _encode(self, texts):
        result = []
        for pos in range(0, len(texts), 32):
            batch = texts[pos : pos + 32]
            if self.provider == "ollama":
                data = self.http.post(
                    "api/embed",
                    {
                        "model": self.model,
                        "input": batch,
                        "truncate": False,
                        "keep_alive": -1,
                    },
                )
                vectors = data.get("embeddings", [])
            else:
                data = self.http.post(
                    "embeddings", {"model": self.model, "input": batch}
                )
                items = sorted(data.get("data", []), key=lambda x: x["index"])
                if [x.get("index") for x in items] != list(range(len(batch))):
                    raise ModelError("embedding indices mismatch")
                vectors = [x["embedding"] for x in items]
            if len(vectors) != len(batch) or any(len(v) != self.dim for v in vectors):
                raise ModelError("embedding shape mismatch")
            result.extend(vectors)
        return result

    def embed_documents(self, texts):
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text):
        return self._encode([self.query_prefix + text])[0]
