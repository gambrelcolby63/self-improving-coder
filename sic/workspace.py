"""The agent's own persistent "computer": a workspace directory owned by the
unprivileged sandbox user, plus tools confined to it.

Tools: run_command, read_file, write_file, list_dir, run_tests.
* In "user" mode (UserSandbox) every tool executes AS the sandbox user inside
  fresh namespaces, so the kernel enforces confinement (other users' homes,
  /workspace, /cursor etc. are hidden; no sudo; scrubbed env).  Shell commands get
  network access (pip/git) but nftables blocks local/private/metadata ranges.
* In "local" mode (CI / no sudo) paths are still validated, but it is NOT a
  security boundary.
Files persist between chats because the directory persists.
"""
from __future__ import annotations

import json
import os
import posixpath
import shlex

from .sandbox import LocalSandbox, UserSandbox, default_sandbox


class PathError(ValueError):
    pass


def safe_relpath(path: str) -> str:
    """Normalise a user/model supplied path to a relative path inside the workspace."""
    if not isinstance(path, str) or "\x00" in path:
        raise PathError("invalid path")
    p = path.strip() or "."
    if p.startswith("~"):
        raise PathError("home-relative paths are not allowed")
    if posixpath.isabs(p):
        raise PathError(f"absolute path not allowed: {path}")
    norm = posixpath.normpath(p)
    if norm == ".." or norm.startswith("../"):
        raise PathError(f"path escapes workspace: {path}")
    return norm


_READ = r'''
import os, sys, json
root = os.path.realpath(os.getcwd())
p = os.path.realpath(sys.argv[1])
if p != root and not p.startswith(root + os.sep):
    print(json.dumps({"error": "path escapes workspace (symlink?)"})); sys.exit(3)
mode = sys.argv[2]
if mode == "read":
    with open(p, errors="replace") as f:
        data = f.read(int(sys.argv[3]) + 1)
    print(json.dumps({"content": data[:int(sys.argv[3])], "truncated": len(data) > int(sys.argv[3])}))
elif mode == "list":
    out = []
    for name in sorted(os.listdir(p))[:500]:
        fp = os.path.join(p, name)
        out.append({"name": name + ("/" if os.path.isdir(fp) else ""), "size": os.path.getsize(fp) if os.path.isfile(fp) else None})
    print(json.dumps({"entries": out}))
elif mode == "write":
    d = os.path.dirname(p)
    os.makedirs(d, exist_ok=True)
    with open(p, "w") as f:
        f.write(open(sys.argv[3]).read())
    os.remove(sys.argv[3])
    print(json.dumps({"written": os.path.relpath(p, root), "bytes": os.path.getsize(p)}))
'''


class AgentComputer:
    def __init__(self, sandbox=None, root: str | None = None, cmd_timeout: int = 60):
        self.sb = sandbox or default_sandbox()
        if isinstance(self.sb, UserSandbox):
            self.root = root or os.path.join(self.sb.home, "ws")
        else:
            self.root = root or os.path.expanduser("~/.sic/agent_ws")
            os.makedirs(self.root, exist_ok=True)
        self.cmd_timeout = cmd_timeout

    # all execution happens inside the sandbox, cwd = workspace root
    def _exec(self, cmd, files=None, timeout=None, network=False):
        return self.sb.run(cmd, files=files, timeout=timeout or self.cmd_timeout, cwd=self.root,
                           network=network)

    def run_command(self, command: str, timeout: int | None = None) -> dict:
        t = min(int(timeout or self.cmd_timeout), 300)
        # pip installs go to the agent user's own ~/.local (never system-wide)
        r = self._exec(["bash", "-c", "export PIP_BREAK_SYSTEM_PACKAGES=1\n" + command], timeout=t, network=True)
        return {"exit": r.returncode, "timed_out": r.timed_out, "stdout": r.stdout[-6000:],
                "stderr": r.stderr[-4000:], "error": r.error}

    def _helper(self, args, files=None):
        r = self._exec(["python3", "-I", ".sic_helper.py"] + args,
                       files={".sic_helper.py": _READ, **(files or {})}, timeout=20)
        self._exec(["rm", "-f", ".sic_helper.py"], timeout=10)
        if r.error:
            return {"error": r.error}
        try:
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            return {"error": (r.stderr or r.stdout)[-500:] or "failed"}

    def read_file(self, path: str, max_chars: int = 20000) -> dict:
        rel = safe_relpath(path)
        return self._helper([rel, "read", str(max_chars)])

    def list_dir(self, path: str = ".") -> dict:
        rel = safe_relpath(path)
        return self._helper([rel, "list"])

    def write_file(self, path: str, content: str) -> dict:
        rel = safe_relpath(path)
        if len(content) > 1_000_000:
            return {"error": "file too large (1 MB cap)"}
        tmp = ".sic_upload.tmp"
        return self._helper([rel, "write", tmp], files={tmp: content})

    def run_tests(self, path: str = ".", timeout: int = 120) -> dict:
        rel = safe_relpath(path)
        cmd = (f"cd {shlex.quote(rel)} && if python3 -c 'import pytest' 2>/dev/null; "
               "then python3 -m pytest -q 2>&1; else python3 -m unittest discover -v 2>&1; fi")
        r = self._exec(["bash", "-c", cmd], timeout=timeout)
        return {"exit": r.returncode, "passed": r.ok, "output": (r.stdout + r.stderr)[-6000:]}

    TOOLS = {
        "run_command": "run_command(command, timeout=60): run a bash command in your workspace (network allowed: pip install --user, git clone).",
        "read_file": "read_file(path): read a text file from your workspace (relative path).",
        "write_file": "write_file(path, content): create/overwrite a file in your workspace.",
        "list_dir": "list_dir(path='.'): list a directory in your workspace.",
        "run_tests": "run_tests(path='.'): run pytest (or unittest) in a workspace folder.",
    }

    def call(self, name: str, args: dict) -> dict:
        if name not in self.TOOLS:
            return {"error": f"unknown tool {name}"}
        try:
            return getattr(self, name)(**args)
        except PathError as e:
            return {"error": f"denied: {e}"}
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}
