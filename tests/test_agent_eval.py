import json

import pytest

from evals.custom_tasks import ALL as CUSTOM
from sic.agent import Task, solve, visible_check
from sic.llm import MockLLM
from sic.runners import run_task, toolchains
from sic.sandbox import LocalSandbox

PROMPT = 'def add(a, b):\n    """Add two numbers.\n    >>> add(2, 3)\n    5\n    """\n'
HIDDEN = "def check(candidate):\n    assert candidate(-1, 1) == 0\n    assert candidate(10, 5) == 15\n"


def task():
    return Task("demo/add", "python", PROMPT, HIDDEN, "add")


def test_repair_loop_fixes_bug_using_visible_feedback():
    llm = MockLLM(["```python\ndef add(a, b):\n    return a - b\n```",
                   "```python\ndef add(a, b):\n    return a + b\n```"])
    res = solve(task(), llm, LocalSandbox(), max_attempts=3)
    assert len(res.attempts) == 2 and res.attempts[0].visible_ok is False and res.visible_ok
    assert "Failed example" in llm.prompts[1][-1]["content"]
    # hidden tests are never shown to the model
    assert all("candidate(-1, 1)" not in m["content"] for p in llm.prompts for m in p)
    assert run_task(LocalSandbox(), "python", res.code, HIDDEN, "add").ok


def test_bare_mode_is_single_call_without_execution():
    llm = MockLLM(["```python\ndef add(a, b):\n    return a - b\n```", "never"])
    res = solve(task(), llm, sandbox=None, max_attempts=1)
    assert llm.calls == 1 and res.attempts[0].feedback == "not run"


def test_attempts_are_bounded():
    llm = MockLLM(fn=lambda m: "```python\ndef add(a, b):\n    return 0\n```")
    res = solve(task(), llm, LocalSandbox(), max_attempts=3)
    assert llm.calls == 3 and not res.visible_ok


def test_visible_check_ignores_closing_docstring_quotes():
    ok, score, _ = visible_check(LocalSandbox(), task(), "def add(a, b):\n    return a + b\n")
    assert ok and score == 1.0


def test_experience_few_shots_are_injected(tmp_path):
    from sic.experience import ExperienceLibrary
    lib = ExperienceLibrary(str(tmp_path / "l.jsonl"))
    lib.add("train/sum", "python", "Add two numbers a and b", "def s(a, b):\n    return a + b\n", 1)
    llm = MockLLM(["```python\ndef add(a, b):\n    return a + b\n```"])
    solve(task(), llm, LocalSandbox(), experience=lib, k_shots=2)
    roles = [m["role"] for m in llm.prompts[0]]
    assert roles[:3] == ["system", "user", "assistant"] and "def s(a, b)" in llm.prompts[0][2]["content"]


@pytest.mark.parametrize("t", CUSTOM, ids=[t["id"] for t in CUSTOM])
def test_custom_task_references_pass(t):
    if not toolchains()[t["lang"]]:
        pytest.skip("toolchain missing")
    sb = LocalSandbox()
    assert run_task(sb, t["lang"], t["ref"], t["visible"]).ok
    assert run_task(sb, t["lang"], t["ref"], t["hidden"]).ok


def test_splits_disjoint_and_library_has_no_eval_tasks():
    import os
    from evals.run_eval import LIB_FROZEN, LIB_TRAIN, SPLITS
    s = json.load(open(SPLITS))
    assert not set(s["eval"]) & set(s["train"])
    for p in (LIB_TRAIN, LIB_FROZEN):
        if os.path.exists(p):
            ids = {json.loads(l)["id"] for l in open(p) if l.strip()}
            assert not ids & set(s["eval"])
