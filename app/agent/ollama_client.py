"""Local Ollama client via loopback REST API (no ollama Python package).

Endpoints:
  GET  /api/tags
  POST /api/generate
  POST /api/chat   (Phase 2 — messages + optional tools)
  POST /api/embeddings (Phase 4 RAG)
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from app.agent.policy import is_loopback_url

logger = logging.getLogger("STDMS.Agent.Ollama")

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Prefer these chat models when present. Stay in the 7–9B Q4 band (8GB laptop GPU).
_CHAT_MODEL_PREFERENCE = (
    "qwen3.5:9b",
    "qwen3:8b",
    "qwen2.5:7b",
    "qwen2.5:8b",
    "qwen2:8b",
    "qwen3:4b",
    "qwen2:7b",
)


def list_ollama_models(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 2.0) -> list[str]:
    """Return installed Ollama model names (empty if offline)."""
    if not is_loopback_url(base_url):
        return []
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/tags",
            method="GET",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        models = []
        for m in data.get("models") or []:
            name = (m.get("name") or m.get("model") or "").strip()
            if name:
                models.append(name)
        return models
    except Exception:
        return []


def resolve_chat_model(
    preferred: str = "",
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Pick an installed chat model; fall back to preferred/default name."""
    want = (preferred or DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
    installed = list_ollama_models(base_url)
    if not installed:
        return want
    # Exact match
    if want in installed:
        return want
    # Tag-insensitive match (qwen3:8b vs qwen3:8b-...)
    for name in installed:
        if name == want or name.startswith(want.split(":")[0] + ":"):
            if want.split(":")[0] in name:
                # prefer exact family
                pass
    for cand in (want, *_CHAT_MODEL_PREFERENCE):

        for name in installed:
            if name == cand or name.startswith(cand):
                return name
    # Any qwen*
    for name in installed:
        if "qwen" in name.lower():
            return name
    return installed[0]


def ollama_reachable(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 1.5) -> bool:
    if not is_loopback_url(base_url):
        return False
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/tags",
            method="GET",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> Optional[dict[str, Any]]:
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.info("Ollama POST failed: %s", exc)
        return None
    except Exception as exc:
        logger.info("Ollama POST failed: %s", exc)
        return None


def call_ollama(
    prompt: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.2,
    num_predict: int = 500,
    timeout: float = 90.0,
) -> Optional[str]:
    """POST /api/generate. Returns response text or None on failure."""
    if not prompt or not is_loopback_url(base_url):
        return None
    # Qwen3 defaults to "thinking" mode which often leaves ``response`` empty;
    # disable think so the operator-facing answer lands in ``response``.
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    data = _post_json(f"{base_url.rstrip('/')}/api/generate", payload, timeout=timeout)
    if not data:
        return None
    text = (data.get("response") or "").strip()
    if not text:
        text = (data.get("thinking") or "").strip()
    return text or None


def chat_ollama(
    messages: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.2,
    num_predict: int = 600,
    timeout: float = 120.0,
) -> Optional[dict[str, Any]]:
    """POST /api/chat. Returns assistant message dict (content, tool_calls) or None.

    Message shape matches Ollama chat API. When ``tools`` is provided, the model
    may return ``tool_calls`` for Phase 2 function calling.
    """
    if not messages or not is_loopback_url(base_url):
        return None
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if tools:
        payload["tools"] = tools
    data = _post_json(f"{base_url.rstrip('/')}/api/chat", payload, timeout=timeout)
    if not data:
        return None
    message = data.get("message")
    if isinstance(message, dict):
        # Normalize empty content + thinking fallback
        content = (message.get("content") or "").strip()
        if not content and message.get("thinking"):
            message = dict(message)
            message["content"] = str(message.get("thinking") or "").strip()
        return message
    # Some versions nest differently
    content = (data.get("response") or data.get("content") or "").strip()
    if content:
        return {"role": "assistant", "content": content, "tool_calls": []}
    return None


def embed_ollama(
    text: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_EMBED_MODEL,
    timeout: float = 60.0,
) -> Optional[list[float]]:
    """POST /api/embeddings (fallback /api/embed). Returns vector or None.

    A 404 usually means the embedding model is not pulled:
    ``ollama pull nomic-embed-text``
    """
    if not (text or "").strip() or not is_loopback_url(base_url):
        return None
    data = _post_json(
        f"{base_url.rstrip('/')}/api/embeddings",
        {"model": model, "prompt": text},
        timeout=timeout,
    )
    if not data:
        data = _post_json(
            f"{base_url.rstrip('/')}/api/embed",
            {"model": model, "input": text},
            timeout=timeout,
        )
    if not data:
        logger.warning(
            "Embedding failed for model %r (often missing model). Run: ollama pull %s",
            model,
            model,
        )
        return None
    emb = data.get("embedding")
    if isinstance(emb, list) and emb:
        return [float(x) for x in emb]
    embs = data.get("embeddings")
    if isinstance(embs, list) and embs:
        first = embs[0]
        if isinstance(first, list) and first:
            return [float(x) for x in first]
    return None
