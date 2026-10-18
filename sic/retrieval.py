"""Dependency-free BM25 index with JSON persistence."""
from __future__ import annotations

import json
import math
import re
from collections import Counter

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
STOP = set("the a an and or of to in is it for on with as be this that by are at from return returns "
           "if else def int str list".split())


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOK.findall(text):
        t = t.lower()
        # split snake_case / camelCase pieces as extra tokens
        parts = [p for p in re.split(r"_|(?<=[a-z])(?=[A-Z])", t) if p]
        for p in [t] + (parts if len(parts) > 1 else []):
            if p not in STOP and len(p) > 1:
                out.append(p)
    return out


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[dict] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._len: list[int] = []

    def add(self, text: str, meta: dict | None = None):
        toks = tokenize(text)
        tf = Counter(toks)
        self.docs.append({"text": text, "meta": meta or {}})
        self._tf.append(tf)
        self._len.append(len(toks))
        self._df.update(tf.keys())

    def __len__(self):
        return len(self.docs)

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[tuple[float, dict]]:
        if not self.docs:
            return []
        q = tokenize(query)
        n = len(self.docs)
        avg = sum(self._len) / n or 1.0
        scores = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[i] / avg))
            if s > min_score:
                scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [(s, self.docs[i]) for s, i in scores[:k]]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"docs": self.docs}, f)

    @classmethod
    def load(cls, path: str) -> "BM25":
        idx = cls()
        with open(path) as f:
            for d in json.load(f)["docs"]:
                idx.add(d["text"], d["meta"])
        return idx
