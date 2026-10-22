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
