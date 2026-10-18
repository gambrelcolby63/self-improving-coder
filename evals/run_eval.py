"""Resumable evaluation + learning round + promotion gate.

Conditions
  a      bare model, single shot, no execution feedback
  b      agent: sandbox run on VISIBLE checks + up to 3 repair attempts
  learn  learning round on the TRAIN split (agent b + growing experience library);
         only solutions that pass the train task's full tests are stored
  c      agent b + frozen experience library (few-shot retrieval), on the EVAL split

Pass@1 = final code (one per task, greedy decoding temperature 0 / top_k 1, seed 0)
passes the hidden tests.  Every per-task result is appended to results/<cond>.jsonl
immediately, so an interrupted run resumes where it stopped.

    python -m evals.run_eval --cond a
    python -m evals.run_eval --summary
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evals.custom_tasks import ALL as CUSTOM  # noqa: E402
from sic.agent import SYSTEM, Task, solve  # noqa: E402
from sic.experience import ExperienceLibrary  # noqa: E402
from sic.runners import run_task, toolchains  # noqa: E402

RES = os.path.join(ROOT, "results")
HE_PATH = os.path.join(ROOT, "data", "HumanEval.jsonl.gz")
SPLITS = os.path.join(ROOT, "evals", "splits.json")
LIB_TRAIN = os.path.join(ROOT, "store", "experience_train.jsonl")
LIB_FROZEN = os.path.join(ROOT, "store", "experience_frozen.jsonl")
N_EVAL_HE = 80
SEED = 1234


def load_humaneval():
    with gzip.open(HE_PATH, "rt") as f:
        return [json.loads(l) for l in f]


def make_splits():
    if os.path.exists(SPLITS):
        with open(SPLITS) as f:
            return json.load(f)
    ids = sorted(p["task_id"] for p in load_humaneval())
    random.Random(SEED).shuffle(ids)
    tc = toolchains()
    custom = [t for t in CUSTOM if tc[t["lang"]]]
    s = {"seed": SEED,
         "eval": sorted(ids[:N_EVAL_HE], key=lambda x: int(x.split("/")[1])) +
                 [t["id"] for t in custom if t["split"] == "eval"],
         "train": sorted(ids[N_EVAL_HE:], key=lambda x: int(x.split("/")[1])) +
                  [t["id"] for t in custom if t["split"] == "train"]}
    assert not set(s["eval"]) & set(s["train"])
    with open(SPLITS, "w") as f:
        json.dump(s, f, indent=1)
    return s


def all_tasks() -> dict[str, Task]:
    out = {}
    for p in load_humaneval():
        out[p["task_id"]] = Task(p["task_id"], "python", p["prompt"], p["test"], p["entry_point"],
                                 meta={"bench": "HumanEval"})
    for t in CUSTOM:
        out[t["id"]] = Task(t["id"], t["lang"], t["prompt"], t["hidden"], None, t["visible"],
                            meta={"bench": "custom-" + t["lang"]})
    return out


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(l)["task_id"] for l in f if l.strip()}


def score(sandbox, task, code) -> tuple[bool, str]:
    r = run_task(sandbox, task.lang, code, task.test, task.entry_point, timeout=15)
    return r.ok, r.summary(600)


def run(cond: str, llm, sandbox, limit: int | None = None):
    os.makedirs(RES, exist_ok=True)
    splits = make_splits()
    tasks = all_tasks()
    ids = splits["train" if cond == "learn" else "eval"][:limit]
    out_path = os.path.join(RES, f"{cond}.jsonl")
    done = done_ids(out_path)
    lib = None
    if cond == "learn":
        lib = ExperienceLibrary(LIB_TRAIN)
    elif cond == "c":
        if not os.path.exists(LIB_FROZEN):
            shutil.copy(LIB_TRAIN, LIB_FROZEN)
        lib = ExperienceLibrary(LIB_FROZEN)
        assert not {it["id"] for it in lib.items} & set(splits["eval"]), "eval task leaked into library"
    for n, tid in enumerate(ids):
        if tid in done:
            continue
        t = tasks[tid]
        t0 = time.time()
        if cond == "a":
            res = solve(t, llm, sandbox=None, max_attempts=1)
        else:
            res = solve(t, llm, sandbox=sandbox, max_attempts=3, experience=lib, k_shots=2 if lib else 0)
        passed, log = score(sandbox, t, res.code)
        if cond == "learn" and passed:
            failed = [{"code": a.code, "feedback": a.feedback} for a in res.attempts if not a.visible_ok]
            lib.add(tid, t.lang, t.user_message(), res.code, len(res.attempts), failed, t.meta["bench"])
        rec = {"task_id": tid, "lang": t.lang, "bench": t.meta["bench"], "passed": passed,
               "visible_ok": res.visible_ok if cond != "a" else None, "attempts": len(res.attempts),
               "llm_calls": res.llm_calls, "seconds": round(time.time() - t0, 1),
               "code": res.code, "hidden_log": "" if passed else log[-400:]}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{cond}] {n + 1}/{len(ids)} {tid} passed={passed} attempts={len(res.attempts)} "
              f"{rec['seconds']}s", flush=True)
    if cond == "learn":
        n = lib.export_jsonl(os.path.join(ROOT, "data", "finetune", "experience_sft.jsonl"), SYSTEM["python"])
        print(f"exported {n} SFT records")


def load(cond):
    p = os.path.join(RES, f"{cond}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize():
    splits = make_splits()
    rows, summ = [], {}
    for cond in ["a", "b", "c", "learn"]:
        recs = load(cond)
        if not recs:
            continue
        groups = {}
        for r in recs:
            groups.setdefault(r["bench"], []).append(r["passed"])
            groups.setdefault("ALL", []).append(r["passed"])
        summ[cond] = {g: {"n": len(v), "passed": sum(v), "pass@1": round(100 * sum(v) / len(v), 1)}
                      for g, v in groups.items()}
        summ[cond]["_meta"] = {"llm_calls": sum(r["llm_calls"] for r in recs),
                               "minutes": round(sum(r["seconds"] for r in recs) / 60, 1),
                               "complete": len(recs) == len(splits["train" if cond == "learn" else "eval"])}
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    write_markdown(summ)
    write_svg(summ)
    promote(summ)
    return summ


LABEL = {"a": "(a) bare model, single-shot", "b": "(b) agent + sandbox repair",
         "c": "(c) agent + repair + experience library", "learn": "learning round (TRAIN split)"}


def write_markdown(summ):
    benches = sorted({g for c in summ.values() for g in c if not g.startswith("_")} - {"ALL"}) + ["ALL"]
    lines = ["| condition | " + " | ".join(benches) + " | LLM calls | wall min |",
             "|---|" + "---|" * (len(benches) + 2)]
    for cond, s in summ.items():
        cells = [f"{s[b]['pass@1']}% ({s[b]['passed']}/{s[b]['n']})" if b in s else "-" for b in benches]
        lines.append(f"| {LABEL[cond]} | " + " | ".join(cells) +
                     f" | {s['_meta']['llm_calls']} | {s['_meta']['minutes']} |")
    with open(os.path.join(RES, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_svg(summ):
    conds = [c for c in ["a", "b", "c"] if c in summ]
    benches = sorted({g for c in conds for g in summ[c] if not g.startswith("_")} - {"ALL"}) + ["ALL"]
    colors = {"a": "#9aa5b1", "b": "#3b82f6", "c": "#16a34a"}
    W, H, pad, bw = 120 * len(benches) + 80, 300, 40, 28
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H + 60}" font-family="sans-serif" font-size="11">',
           f'<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{pad}" y="16" font-size="13">pass@1 (%) on held-out eval split</text>']
    for i, b in enumerate(benches):
        x0 = pad + i * 120
        for j, c in enumerate(conds):
            v = summ[c].get(b, {}).get("pass@1", 0)
            h = v / 100 * (H - 60)
            x = x0 + j * (bw + 4)
            out.append(f'<rect x="{x}" y="{H - h}" width="{bw}" height="{h}" fill="{colors[c]}"/>')
            out.append(f'<text x="{x + bw / 2}" y="{H - h - 3}" text-anchor="middle">{v:.0f}</text>')
        n = summ[conds[0]].get(b, {}).get("n", 0)
        out.append(f'<text x="{x0 + 40}" y="{H + 16}" text-anchor="middle">{b} (n={n})</text>')
    for j, c in enumerate(conds):
        out.append(f'<rect x="{pad + j * 230}" y="{H + 30}" width="10" height="10" fill="{colors[c]}"/>'
                   f'<text x="{pad + j * 230 + 14}" y="{H + 39}">{LABEL[c]}</text>')
    out.append("</svg>")
    with open(os.path.join(ROOT, "docs", "results.svg"), "w") as f:
        f.write("\n".join(out))


def promote(summ):
    """Promotion gate: a candidate becomes champion only if it strictly beats the champion's
    overall pass@1 on the same, complete eval split."""
    path = os.path.join(RES, "champion.json")
    champ = json.load(open(path)) if os.path.exists(path) else None
    log = []
    for cond in ["a", "b", "c"]:
        if cond not in summ or not summ[cond]["_meta"]["complete"]:
            continue
        sc = summ[cond]["ALL"]["pass@1"]
        lib_hash = None
        if cond == "c" and os.path.exists(LIB_FROZEN):
            lib_hash = hashlib.sha256(open(LIB_FROZEN, "rb").read()).hexdigest()[:12]
        if champ is None or sc > champ["pass@1"]:
            log.append(f"promote {cond} ({sc}%) over {champ['version'] if champ else 'none'}"
                       f" ({champ['pass@1'] if champ else '-'}%)")
            champ = {"version": cond, "label": LABEL[cond], "pass@1": sc, "experience_lib_sha": lib_hash,
                     "n": summ[cond]["ALL"]["n"]}
        else:
            log.append(f"reject {cond} ({sc}%): does not beat champion {champ['version']} ({champ['pass@1']}%)")
    if champ:
        champ["gate_log"] = log
        with open(path, "w") as f:
            json.dump(champ, f, indent=1)
    for l in log:
        print(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["a", "b", "learn", "c"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.cond:
        from sic.llm import OllamaLLM
        from sic.sandbox import default_sandbox
        sb = default_sandbox()
        print("sandbox:", sb.kind, flush=True)
        run(a.cond, OllamaLLM(), sb, a.limit)
    if a.summary or not a.cond:
        print(json.dumps(summarize(), indent=1))


if __name__ == "__main__":
    main()
