"""Write -> run -> read errors -> repair loop.

The agent only ever sees *visible* checks (doctest examples from the prompt, or a
task's `visible_test`) plus compiler/runtime errors.  Hidden benchmark tests are
used by the evaluator for scoring and never shown to the model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .codeutil import extract_code
from .runners import run_task

SYSTEM = {
    "python": "You are an expert Python programmer. Write correct, self-contained code. "
              "Reply with exactly one ```python code block containing the complete solution.",
    "cpp": "You are an expert C++17 programmer. Reply with exactly one ```cpp code block. "
           "Include the headers you need. Do NOT write a main() function.",
    "verilog": "You are an expert digital designer. Write synthesizable Verilog-2005 (iverilog compatible). "
               "Reply with exactly one ```verilog code block containing only the requested module(s); "
               "no testbench.",
}
FENCE = {"python": "python", "cpp": "cpp", "verilog": "verilog"}

_VISIBLE_PY = r'''
import doctest, sys
CODE = {code!r}
DOC = {doc!r}
ns = {{}}
exec(compile(CODE, "solution.py", "exec"), ns)
ex = doctest.DocTestParser().get_examples(DOC)
if not ex:
    print("VISIBLE 0/0 (no examples; code compiled and imported fine)")
    sys.exit(0)
t = doctest.DocTest(ex, ns, "visible", "solution.py", 0, None)
r = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
r.run(t)
print(f"VISIBLE {{r.tries - r.failures}}/{{r.tries}}")
sys.exit(1 if r.failures else 0)
'''


@dataclass
class Task:
    id: str
    lang: str
    prompt: str
    test: str  # hidden test (scoring only)
    entry_point: str | None = None
    visible_test: str | None = None  # shown-to-agent check (cpp/verilog); python uses doctests
    meta: dict = field(default_factory=dict)

    def user_message(self) -> str:
        if self.lang == "python" and self.entry_point:
            return ("Complete the following Python function. Return the full implementation "
                    "(including the imports and the def line).\n\n```python\n" + self.prompt + "```")
        return self.prompt


@dataclass
class Attempt:
    code: str
    visible_ok: bool
    visible_score: float
    feedback: str


@dataclass
class SolveResult:
    task_id: str
    code: str
    attempts: list[Attempt]
    llm_calls: int

    @property
    def visible_ok(self) -> bool:
        return any(a.visible_ok for a in self.attempts)


def visible_check(sandbox, task: Task, code: str) -> tuple[bool, float, str]:
    if task.lang == "python":
        doc = re.sub(r'(?m)^\s*("""|\'\'\')\s*$', "", task.prompt)  # closing quotes are not output
        prog = _VISIBLE_PY.format(code=code, doc=doc)
        r = sandbox.run(["python3", "-I", "visible.py"], files={"visible.py": prog}, timeout=10)
        score = 0.0
        for line in r.stdout.splitlines():
            if line.startswith("VISIBLE "):
                a, b = line.split()[1].split("/")
                score = (int(a) / int(b)) if int(b) else 0.5
        if r.timed_out:
            r.stderr += "\nTimed out (possible infinite loop)."
        return r.ok, score if r.ok else min(score, 0.99), r.summary()
    r = run_task(sandbox, task.lang, code, task.visible_test or "", timeout=10)
    return r.ok, 1.0 if r.ok else 0.0, r.summary()


def few_shot_messages(examples: list[dict]) -> list[dict]:
    msgs = []
    for ex in examples:
        msgs.append({"role": "user", "content": ex["problem"]})
        msgs.append({"role": "assistant", "content": f"```{FENCE[ex['lang']]}\n{ex['solution'].strip()}\n```"})
    return msgs


def solve(task: Task, llm, sandbox=None, max_attempts: int = 3, experience=None, k_shots: int = 2,
          docs=None, k_docs: int = 0, on_event=None) -> SolveResult:
    """max_attempts=1 and sandbox=None gives the bare single-shot baseline."""
    calls0 = llm.calls
    user = task.user_message()
    msgs = [{"role": "system", "content": SYSTEM[task.lang]}]
    if experience is not None and k_shots:
        shots = experience.retrieve(user, k=k_shots, lang=task.lang, exclude_id=task.id)
        msgs += few_shot_messages(shots)
        if on_event:
            on_event({"type": "experience", "ids": [s["id"] for s in shots]})
    if docs is not None and k_docs:
        hits = docs.search(user, k=k_docs)
        if hits:
            ctx = "\n\n".join(f"[source: {d['meta'].get('source')}]\n{d['text'][:1200]}" for _, d in hits)
            user = "Reference snippets (may be irrelevant):\n" + ctx + "\n\n---\n" + user
    msgs.append({"role": "user", "content": user})
    attempts: list[Attempt] = []
    for i in range(max_attempts):
        reply = llm.chat(msgs)
        code = extract_code(reply, task.lang)
        if sandbox is None:
            attempts.append(Attempt(code, False, 0.0, "not run"))
            break
        ok, score, fb = visible_check(sandbox, task, code)
        attempts.append(Attempt(code, ok, score, fb))
        if on_event:
            on_event({"type": "attempt", "n": i + 1, "ok": ok, "feedback": fb[:800]})
        if ok:
            break
        msgs.append({"role": "assistant", "content": f"```{FENCE[task.lang]}\n{code.strip()}\n```"})
        msgs.append({"role": "user", "content": "The sandbox run failed:\n" + fb[:2000] +
                     "\n\nFix the bug. Reply with the full corrected code in one code block."})
    best = max(attempts, key=lambda a: (a.visible_ok, a.visible_score)) if sandbox else attempts[-1]
    # tie-break: max() keeps the earliest among equals
    return SolveResult(task.id, best.code, attempts, llm.calls - calls0)


def to_json(res: SolveResult) -> str:
    return json.dumps({"task_id": res.task_id, "code": res.code, "llm_calls": res.llm_calls,
                       "attempts": [a.__dict__ for a in res.attempts]})
