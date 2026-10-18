import json
import os

import pytest

from sic.codeutil import extract_code, strip_main_guard
from sic.experience import ExperienceLibrary
from sic.retrieval import BM25, tokenize
from sic.runners import run_task, toolchains
from sic.sandbox import LocalSandbox


def test_extract_code_prefers_language_block():
    t = "intro\n```text\nnope\n```\n```python\ndef f():\n    return 1\n```\n"
    assert extract_code(t, "python").startswith("def f()")


def test_extract_code_unterminated_and_raw():
    assert "x = 1" in extract_code("```python\nx = 1\n", "python")
    assert extract_code("x = 2", "python").strip() == "x = 2"


def test_strip_main_guard():
    code = "def f():\n    return 1\n\nif __name__ == '__main__':\n    print(f())\n    input()\n"
    out = strip_main_guard(code)
    assert "input()" not in out and "def f" in out


def test_bm25_ranks_relevant_first(tmp_path):
    idx = BM25()
    idx.add("sort a list of integers using merge sort", {"id": 1})
    idx.add("verilog counter with synchronous reset", {"id": 2})
    idx.add("parse json from a string", {"id": 3})
    assert idx.search("synchronous reset counter")[0][1]["meta"]["id"] == 2
    p = tmp_path / "s.json"
    idx.save(str(p))
    assert BM25.load(str(p)).search("merge sort")[0][1]["meta"]["id"] == 1
    assert "snake" in tokenize("snake_case_name")


def test_experience_library_roundtrip(tmp_path):
    lib = ExperienceLibrary(str(tmp_path / "lib.jsonl"))
    lib.add("t1", "python", "reverse a string", "def r(s):\n    return s[::-1]\n", 2,
            [{"code": "def r(s): return s", "feedback": "AssertionError"}])
    lib.add("t1", "python", "dup", "x", 1)  # duplicate id ignored
    lib.add("t2", "cpp", "reverse a string in c++", "std::string r();", 1)
    lib2 = ExperienceLibrary(str(tmp_path / "lib.jsonl"))
    assert len(lib2.items) == 2
    assert lib2.retrieve("reverse string", lang="python")[0]["id"] == "t1"
    assert lib2.retrieve("reverse string", exclude_id="t1", lang="python") == []
    n = lib2.export_jsonl(str(tmp_path / "sft.jsonl"), "sys")
    rows = [json.loads(l) for l in open(tmp_path / "sft.jsonl")]
    assert n == 3 and rows[0]["messages"][-1]["role"] == "assistant"
    assert any(r["meta"]["kind"] == "repair" and len(r["messages"]) == 5 for r in rows)


def test_local_sandbox_basic_timeout_and_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "should-not-leak")
    sb = LocalSandbox()
    r = sb.run(["python3", "-c", "import os;print(os.environ.get('GITHUB_TOKEN'))"])
    assert r.ok and r.stdout.strip() == "None"
    r = sb.run(["python3", "-c", "while True: pass"], timeout=1)
    assert r.timed_out and not r.ok
    r = sb.run(["python3", "-c", "x = bytearray(4 * 1024**3)"], timeout=5)
    assert not r.ok


def test_run_task_python_humaneval_style():
    test = "def check(candidate):\n    assert candidate(2) == 4\n"
    sb = LocalSandbox()
    assert run_task(sb, "python", "def sq(x):\n    return x*x\n", test, "sq").ok
    assert not run_task(sb, "python", "def sq(x):\n    return x+x+1\n", test, "sq").ok


@pytest.mark.skipif(not toolchains()["cpp"], reason="g++ missing")
def test_run_task_cpp():
    sb = LocalSandbox()
    t = "#include <cassert>\nint main(){assert(add(1,2)==3);}"
    assert run_task(sb, "cpp", "int add(int a,int b){return a+b;}", t).ok
    assert not run_task(sb, "cpp", "int add(int a,int b){return a-b;}", t).ok
    assert not run_task(sb, "cpp", "int add(int a,int b){return a+b", t).ok


@pytest.mark.skipif(not toolchains()["verilog"], reason="iverilog missing")
def test_run_task_verilog_requires_pass_marker():
    sb = LocalSandbox()
    tb = 'module tb; reg a; wire y; inv u(a,y); initial begin a=0; #1 if(y!==1) $display("FAIL"); else $display("ALL TESTS PASSED"); end endmodule'
    assert run_task(sb, "verilog", "module inv(input a, output y); assign y=~a; endmodule", tb).ok
    assert not run_task(sb, "verilog", "module inv(input a, output y); assign y=a; endmodule", tb).ok
