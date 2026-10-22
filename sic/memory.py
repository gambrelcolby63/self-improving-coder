"""Tiny long-term memory: facts Colby tells the assistant, kept in state/memory.json (gitignored)."""
from __future__ import annotations

import json
import os
import re
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_FILE = os.environ.get("SIC_MEMORY_FILE", os.path.join(ROOT, "state", "memory.json"))
MAX_FACTS = 60

_PATTERNS = [
    re.compile(r"\b(?:please\s+)?remember(?: that)?[:,]?\s+(.{3,300})", re.I),
    re.compile(r"\b(?:don'?t|do not) forget(?: that)?[:,]?\s+(.{3,300})", re.I),
    re.compile(r"\b(my name is\s+.{2,60})", re.I),
    re.compile(r"\b(call me\s+.{2,40})", re.I),
    re.compile(r"\b(i (?:really )?(?:prefer|love|like|hate|dislike|use|work (?:on|at|as)|live in|am from)\s+.{2,200})", re.I),
    re.compile(r"\b(my (?:favorite|favourite|wife|husband|girlfriend|boyfriend|dog|cat|job|birthday|project|email|phone|timezone)\b.{2,200})", re.I),
]
_FORGET = re.compile(r"\bforget (?:about |that )?(.{3,200})", re.I)


def _clean(s: str) -> str:
    s = re.split(r"(?<=[.!?])\s|\n", s.strip())[0]
    return s.rstrip(" .!?,;").strip()[:300]


class Memory:
    def __init__(self, path: str = MEMORY_FILE):
        self.path = path
        self.lock = threading.Lock()
        self.facts: list[dict] = []
        try:
            with open(path) as f:
                self.facts = json.load(f).get("facts", [])
        except (OSError, ValueError):
            pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"facts": self.facts}, f, indent=1)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def add(self, text: str) -> bool:
        text = _clean(text)
        if len(text) < 3 or any(f["text"].lower() == text.lower() for f in self.facts):
            return False
        with self.lock:
            self.facts.append({"text": text, "t": time.time()})
            self.facts = self.facts[-MAX_FACTS:]
            self._save()
        return True

    def forget(self, text: str) -> int:
        words = {w for w in re.findall(r"\w{3,}", text.lower())}
        keep = [f for f in self.facts if not (words and words <= set(re.findall(r"\w{3,}", f["text"].lower())))]
        n = len(self.facts) - len(keep)
        if n:
            with self.lock:
                self.facts = keep
                self._save()
        return n

    def observe(self, message: str) -> list[str]:
        """Store facts stated in a user message. Returns the newly stored facts."""
        m = _FORGET.search(message)
        if m and not re.search(r"\bdon'?t forget|do not forget", message, re.I):
            self.forget(_clean(m.group(1)))
            return []
        new = []
        # implicit facts only from statements, never from questions ("what's my favorite snack?")
        statements = " ".join(x for x in re.split(r"(?<=[.!?\n])\s*", message)
                              if x.strip() and not x.strip().endswith("?"))
        for p in _PATTERNS:
            for m in p.finditer(message if p in _PATTERNS[:2] else statements):
                fact = m.group(1)
                if p is _PATTERNS[0] or p is _PATTERNS[1]:
                    fact = re.sub(r"^(?:that\s+)?i\b", "Colby", fact.strip(), flags=re.I)
                    fact = re.sub(r"^my\b", "Colby's", fact, flags=re.I)
                else:
                    fact = "Colby said: " + _clean(fact)
                if self.add(fact):
                    new.append(_clean(fact))
            if new and p in _PATTERNS[:2]:
                break  # an explicit "remember" wins; don't double-store fragments
        return new

    def prompt(self) -> str:
        if not self.facts:
            return ""
        return "Things you remember about Colby (from earlier chats):\n" + "\n".join(
            "- " + f["text"] for f in self.facts[-40:])
