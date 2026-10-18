"""Experience library: verified solutions + repair traces, retrieved as few-shot examples
and exported as chat-format JSONL for LoRA fine-tuning."""
from __future__ import annotations

import json
import os
import time

from .retrieval import BM25


class ExperienceLibrary:
    def __init__(self, path: str):
        self.path = path
        self.items: list[dict] = []
        if os.path.exists(path):
            with open(path) as f:
                self.items = [json.loads(l) for l in f if l.strip()]
        self._reindex()

    def _reindex(self):
        self.idx = BM25()
        for it in self.items:
            self.idx.add(it["problem"], {"id": it["id"]})

    def add(self, task_id: str, lang: str, problem: str, solution: str, attempts: int,
            failed_attempts: list[dict] | None = None, source: str = ""):
        """Only call with sandbox-VERIFIED solutions."""
        if any(it["id"] == task_id for it in self.items):
            return
        it = {"id": task_id, "lang": lang, "problem": problem, "solution": solution,
              "attempts": attempts, "failed_attempts": failed_attempts or [], "source": source,
              "ts": int(time.time())}
        self.items.append(it)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(it) + "\n")
        self.idx.add(problem, {"id": task_id})

    def retrieve(self, problem: str, k: int = 2, lang: str | None = None, exclude_id: str | None = None):
        by_id = {it["id"]: it for it in self.items}
        out = []
        for _, d in self.idx.search(problem, k=k + 5):
            it = by_id[d["meta"]["id"]]
            if it["id"] == exclude_id or (lang and it["lang"] != lang):
                continue
            out.append(it)
            if len(out) >= k:
                break
        return out

    def export_jsonl(self, out_path: str, system: str) -> int:
        """Chat-format SFT records. Repair traces become multi-turn (bad code -> error -> fix)."""
        n = 0
        with open(out_path, "w") as f:
            for it in self.items:
                user = it["problem"]
                fence = {"python": "python", "cpp": "cpp", "verilog": "verilog"}[it["lang"]]
                good = f"```{fence}\n{it['solution'].strip()}\n```"
                f.write(json.dumps({"messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user},
                                                 {"role": "assistant", "content": good}],
                                    "meta": {"id": it["id"], "kind": "solve"}}) + "\n")
                n += 1
                if it["failed_attempts"]:
                    fa = it["failed_attempts"][-1]
                    f.write(json.dumps({"messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": f"```{fence}\n{fa['code'].strip()}\n```"},
                        {"role": "user", "content": "The sandbox run failed:\n" + fa["feedback"][:1500]
                         + "\nFix the code. Reply with the full corrected code."},
                        {"role": "assistant", "content": good}],
                        "meta": {"id": it["id"], "kind": "repair"}}) + "\n")
                    n += 1
        return n
