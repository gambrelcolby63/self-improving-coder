"""Language-specific build/test recipes executed inside a sandbox."""
from __future__ import annotations

import shutil

from .codeutil import strip_main_guard
from .sandbox import RunResult

PASS_MARK = "ALL TESTS PASSED"


def toolchains() -> dict:
    return {"python": True, "cpp": bool(shutil.which("g++")),
            "verilog": bool(shutil.which("iverilog") and shutil.which("vvp"))}


def run_task(sandbox, lang: str, code: str, test: str, entry_point: str | None = None,
             timeout: int = 10) -> RunResult:
    """Run `code` against `test`. Pass == exit 0 (and PASS_MARK for verilog)."""
    if lang == "python":
        prog = strip_main_guard(code) + "\n\n" + test + "\n"
        if entry_point and "check(" in test:
            prog += f"\ncheck({entry_point})\n"
        return sandbox.run(["python3", "-I", "solution_test.py"], files={"solution_test.py": prog},
                           timeout=timeout)
    if lang == "cpp":
        files = {"solution.cpp": code, "test.cpp": '#include "solution.cpp"\n' + test}
        return sandbox.run(["bash", "-c", "g++ -std=c++17 -O1 -o t test.cpp 2>&1 && ./t"], files=files,
                           timeout=timeout + 20)
    if lang == "verilog":
        files = {"design.v": code, "tb.v": test}
        r = sandbox.run(["bash", "-c", "iverilog -g2012 -o sim design.v tb.v 2>&1 && vvp -n sim"],
                        files=files, timeout=timeout + 10)
        if r.returncode == 0 and (PASS_MARK not in r.stdout or "FAIL" in r.stdout):
            r.returncode = 1
            r.stderr += "\ntestbench did not report " + PASS_MARK
        return r
    raise ValueError(f"unknown language {lang}")
