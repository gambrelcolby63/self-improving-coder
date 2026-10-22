"""Minimal Ollama client (stdlib only) + a scripted mock for tests."""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_MODEL = os.environ.get("SIC_MODEL", "sic-qwen2.5-coder-1.5b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


class OllamaLLM:
    def __init__(self, model: str = DEFAULT_MODEL, num_ctx: int = 4096, temperature: float = 0.0,
                 seed: int = 0, num_predict: int = 768, keep_alive: str = "10m", url: str = OLLAMA_URL):
        self.model, self.url = model, url
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx,
                        "num_predict": num_predict, "top_k": 1}
        self.keep_alive = keep_alive
        self.calls = 0
        self.tokens = 0

    def chat(self, messages: list[dict], on_token=None, **opts) -> str:
        """Return the reply. If on_token is given, stream and call it with each text chunk."""
        body = {"model": self.model, "messages": messages, "stream": on_token is not None,
                "options": {**self.options, **opts}, "keep_alive": self.keep_alive}
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        self.calls += 1
        with urllib.request.urlopen(req, timeout=600) as r:
            if on_token is None:
                d = json.loads(r.read())
                self.tokens += d.get("eval_count", 0)
                return d["message"]["content"]
            parts = []
            for line in r:
                if not line.strip():
                    continue
                d = json.loads(line)
                chunk = d.get("message", {}).get("content", "")
                if chunk:
                    parts.append(chunk)
                    on_token(chunk)
                if d.get("done"):
                    self.tokens += d.get("eval_count", 0)
                    break
            return "".join(parts)


class MockLLM:
    """Returns scripted replies in order (or via a function). Records prompts."""

    def __init__(self, replies=None, fn=None):
        self.replies = list(replies or [])
        self.fn = fn
        self.prompts: list[list[dict]] = []
        self.calls = 0
        self.tokens = 0

    def chat(self, messages, on_token=None, **opts) -> str:
        self.prompts.append(list(messages))
        self.calls += 1
        out = self.fn(messages) if self.fn else (self.replies.pop(0) if self.replies else "")
        if on_token and out:
            on_token(out)
        return out


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


class OpenAICompatLLM:
    """Client for an OpenAI-compatible /v1/chat/completions endpoint (vLLM on Modal). Streams via SSE.
    `first_token_timeout` bounds how long we wait for the response to start (cold starts)."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.7,
                 max_tokens: int = 1024, first_token_timeout: float = 25.0, read_timeout: float = 120.0):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.first_token_timeout, self.read_timeout = first_token_timeout, read_timeout
        self.calls = 0
        self.tokens = 0

    def _body(self, messages, stream, opts):
        body = {"model": self.model, "messages": messages, "stream": stream,
                "temperature": opts.get("temperature", self.temperature),
                "max_tokens": int(opts.get("num_predict", opts.get("max_tokens", self.max_tokens)))}
        if opts.get("stop"):
            body["stop"] = list(opts["stop"])[:4]  # OpenAI API allows at most 4
        return body

    def chat(self, messages: list[dict], on_token=None, **opts) -> str:
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(self._body(messages, on_token is not None, opts)).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        self.calls += 1
        # streaming: tokens arrive continuously, so a short per-read timeout detects cold starts fast;
        # non-streaming: nothing arrives until the whole answer is done, so allow the long timeout.
        timeout = self.first_token_timeout if on_token is not None else self.read_timeout
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if on_token is None:
                d = json.loads(r.read())
                self.tokens += d.get("usage", {}).get("completion_tokens", 0)
                return d["choices"][0]["message"]["content"] or ""
            parts = []
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                ch = (d.get("choices") or [{}])[0]
                chunk = (ch.get("delta") or {}).get("content") or ""
                if chunk:
                    parts.append(chunk)
                    on_token(chunk)
            self.tokens += len(parts)
            return "".join(parts)

    def warm(self) -> None:
        """Fire-and-forget request that makes Modal start a container (cold start runs in background)."""
        import threading

        def go():
            try:
                urllib.request.urlopen(self.base_url + "/health", timeout=900).read()
            except Exception:
                pass
        threading.Thread(target=go, daemon=True).start()


class FallbackLLM:
    """Use `primary` (Modal); if it errors or doesn't start answering within its first-token timeout
    (cold start), answer with `fallback` (local Ollama) instead, warm the primary in the background,
    and skip it for `cooldown` seconds. Only falls back if no token has been streamed yet."""

    def __init__(self, primary, fallback, cooldown: float = 90.0):
        self.primary, self.fallback, self.cooldown = primary, fallback, cooldown
        self.down_until = 0.0
        self.last_backend = None
        self.calls = 0

    @property
    def tokens(self):
        return getattr(self.primary, "tokens", 0) + getattr(self.fallback, "tokens", 0)

    def chat(self, messages, on_token=None, **opts) -> str:
        import time
        self.calls += 1
        if time.time() >= self.down_until:
            got = []

            def tok(t):
                got.append(t)
                if on_token:
                    on_token(t)
            try:
                out = self.primary.chat(messages, on_token=tok if on_token else None, **opts)
                self.last_backend = "modal"
                return out
            except Exception:
                if got:  # already streamed part of an answer; don't mix two models' text
                    raise
                self.down_until = time.time() + self.cooldown
                if hasattr(self.primary, "warm"):
                    self.primary.warm()
        self.last_backend = "ollama"
        return self.fallback.chat(messages, on_token=on_token, **opts)


MODAL_CHAT_MODEL = "dolphin3-8b"
MODAL_CODER_MODEL = "qwen2.5-coder-14b"
LOCAL_EQUIV = {MODAL_CHAT_MODEL: "dolphin3-3b", MODAL_CODER_MODEL: "sic-qwen2.5-coder-1.5b"}


def backend() -> str:
    return os.environ.get("SIC_BACKEND", "ollama").lower()


def available_models() -> list[str]:
    local = [m for m in os.environ.get("SIC_MODELS", "dolphin3-3b,sic-qwen2.5-coder-1.5b").split(",") if m]
    if backend() != "modal":
        return local
    remote = [MODAL_CHAT_MODEL] + ([MODAL_CODER_MODEL] if os.environ.get("SIC_MODAL_CODER_URL") else [])
    return remote + local


def make_llm(model: str, num_ctx: int = 8192, temperature: float = 0.7, num_predict: int = 1024):
    """SIC_BACKEND=modal|ollama. Modal models get automatic fallback to the local Ollama equivalent."""
    def local(name):
        llm = OllamaLLM(model=name, num_predict=num_predict, temperature=temperature, num_ctx=num_ctx,
                        keep_alive="30m")
        llm.options["top_k"] = 40
        return llm
    if backend() == "modal" and model in LOCAL_EQUIV:
        url = (os.environ.get("SIC_MODAL_CODER_URL") if model == MODAL_CODER_MODEL
               else os.environ.get("SIC_MODAL_URL") or _read(os.path.join(ROOT, ".modal_url")))
        key = os.environ.get("SIC_MODAL_API_KEY") or _read(os.environ.get(
            "SIC_MODAL_API_KEY_FILE", os.path.join(ROOT, ".modal_api_key")))
        if url and key:
            remote = OpenAICompatLLM(url, key, model, temperature=temperature, max_tokens=num_predict,
                                     first_token_timeout=float(os.environ.get("SIC_MODAL_FIRST_TOKEN_TIMEOUT", "25")))
            return FallbackLLM(remote, local(LOCAL_EQUIV[model]))
        return local(LOCAL_EQUIV[model])
    return local(model)
