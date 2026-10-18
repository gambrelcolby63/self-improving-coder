"""Sandboxed code execution.

Two backends share one tiny bootstrap program (``_BOOTSTRAP``) that receives a
JSON job on stdin ({files, cmd, timeout}), writes the files into its working
directory, runs the command in a new session with a timeout and returns a JSON
result on stdout.

* ``UserSandbox`` (preferred, needs passwordless sudo + scripts/setup_sandbox_user.sh):
  runs the bootstrap as the unprivileged ``sicagent`` user inside fresh mount +
  PID namespaces (optionally a network namespace with no interfaces).  Private
  directories of the host (/workspace, /home/<others>, /cursor, /opt, /tmp, ...)
  are covered with empty tmpfs mounts, /proc is remounted so other processes
  are invisible, the environment is scrubbed (``env -i``) and rlimits are
  applied with ``prlimit``.  Outbound traffic from that uid to loopback /
  private / link-local / metadata ranges is rejected by nftables.
* ``LocalSandbox`` (fallback, e.g. CI): same bootstrap as the current user with
  rlimits, a scrubbed environment and a throwaway temp dir.  This is NOT a
  security boundary; it only keeps honest mistakes contained.

This is a sandbox built from Linux primitives, not a VM.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

AGENT_USER = os.environ.get("SIC_AGENT_USER", "sicagent")
SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"

_BOOTSTRAP = r'''
import json, os, signal, subprocess, sys, time
job = json.loads(sys.stdin.read())
wd = job.get("cwd") or os.getcwd()
os.makedirs(wd, exist_ok=True)
os.chdir(wd)
for name, content in (job.get("files") or {}).items():
    d = os.path.dirname(name)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(name, "w") as f:
        f.write(content)
res = {"returncode": None, "stdout": "", "stderr": "", "timed_out": False}
if job.get("cmd"):
    t0 = time.time()
    p = subprocess.Popen(job["cmd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        out, err = p.communicate(timeout=job.get("timeout", 10))
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        res["timed_out"] = True
    lim = job.get("max_output", 20000)
    res.update(returncode=p.returncode, stdout=out.decode("utf-8", "replace")[-lim:],
               stderr=err.decode("utf-8", "replace")[-lim:], seconds=round(time.time() - t0, 3))
sys.stdout.write("\n__SIC_RESULT__" + json.dumps(res))
'''


@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    seconds: float = 0.0
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None

    def summary(self, limit: int = 1500) -> str:
        parts = [f"exit={self.returncode} timed_out={self.timed_out}"]
        if self.error:
            parts.append(f"sandbox error: {self.error}")
        if self.stdout.strip():
            parts.append("stdout:\n" + self.stdout[-limit:])
        if self.stderr.strip():
            parts.append("stderr:\n" + self.stderr[-limit:])
        return "\n".join(parts)


def _parse(proc_out: str, proc_err: str, rc: int) -> RunResult:
    marker = "__SIC_RESULT__"
    if marker not in proc_out:
        return RunResult(rc, proc_out, proc_err, error=f"bootstrap failed (rc={rc}): {proc_err[-500:]}")
    d = json.loads(proc_out.rsplit(marker, 1)[1])
    return RunResult(d["returncode"], d["stdout"], d["stderr"], d["timed_out"], d.get("seconds", 0.0))


class LocalSandbox:
    """Fallback: current user, scrubbed env, rlimits, temp dir. Not a security boundary."""

    kind = "local"

    def __init__(self, mem_mb: int = 1024, nproc: int = 512, fsize_mb: int = 64):
        self.mem_mb, self.nproc, self.fsize_mb = mem_mb, nproc, fsize_mb

    def _preexec(self, cpu_s: int):
        import resource

        def f():
            os.setsid()
            m = self.mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (m, m))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_FSIZE, (self.fsize_mb << 20, self.fsize_mb << 20))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        return f

    def run(self, cmd, files=None, timeout=10, cwd=None, network=False) -> RunResult:
        tmp = None
        if cwd is None:
            tmp = cwd = tempfile.mkdtemp(prefix="sic_run_")
        env = {"PATH": SAFE_PATH, "HOME": cwd, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
        job = json.dumps({"files": files or {}, "cmd": cmd, "timeout": timeout, "cwd": cwd})
        try:
            p = subprocess.run([sys.executable, "-I", "-c", _BOOTSTRAP], input=job, capture_output=True,
                               text=True, env=env, cwd=cwd, timeout=timeout + 15,
                               preexec_fn=self._preexec(int(timeout) + 5))
            return _parse(p.stdout, p.stderr, p.returncode)
        except subprocess.TimeoutExpired:
            return RunResult(None, "", "", timed_out=True, error="outer timeout")
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)


# Directories hidden (covered by empty tmpfs) inside the user sandbox.
HIDDEN_DIRS = ["/workspace", "/cursor", "/opt", "/exec-daemon", "/tmp", "/var/tmp", "/dev/shm",
               "/run/user", "/srv", "/mnt", "/media", "/root"]

_NS_SCRIPT = r'''
set -e
AGENT_USER="$1"; shift
MEM="$1"; shift
NPROC="$1"; shift
for d in __HIDDEN__; do
  [ -d "$d" ] && mount -t tmpfs -o size=256m,mode=1777 none "$d" 2>/dev/null || true
done
AHOME=$(getent passwd "$AGENT_USER" | cut -d: -f6)
for h in /home/*; do
  [ "$h" = "$AHOME" ] || mount -t tmpfs -o size=1m,mode=755 none "$h" 2>/dev/null || true
done
mount -t proc proc /proc
cd "$AHOME"
exec setpriv --reuid="$AGENT_USER" --regid="$AGENT_USER" --init-groups --inh-caps=-all --no-new-privs \
  env -i PATH=__PATH__ HOME="$AHOME" USER="$AGENT_USER" LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 \
  PIP_USER=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
  prlimit --as="$MEM" --nproc="$NPROC" --fsize=268435456 --core=0 --nofile=1024 -- "$@"
'''.replace("__HIDDEN__", " ".join(HIDDEN_DIRS)).replace("__PATH__", "$AHOME/.local/bin:" + SAFE_PATH)


class UserSandbox:
    """Runs as the dedicated unprivileged user inside mount+PID(+net) namespaces."""

    kind = "user"

    def __init__(self, user: str = AGENT_USER, mem_mb: int = 1536, nproc: int = 256):
        self.user, self.mem_mb, self.nproc = user, mem_mb, nproc
        self.home = os.path.expanduser(f"~{user}")

    @staticmethod
    def available(user: str = AGENT_USER) -> bool:
        if os.environ.get("SIC_FORCE_LOCAL_SANDBOX"):
            return False
        try:
            import pwd
            pwd.getpwnam(user)
        except KeyError:
            return False
        if not shutil.which("sudo") or not shutil.which("unshare") or not shutil.which("setpriv"):
            return False
        return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0

    def run(self, cmd, files=None, timeout=10, cwd=None, network=False) -> RunResult:
        """cwd: path inside the agent home (persistent) or None for a throwaway dir in private /tmp."""
        job = json.dumps({"files": files or {}, "cmd": cmd, "timeout": timeout,
                          "cwd": cwd or "/tmp/run"})
        ns = ["unshare", "--mount", "--pid", "--fork", "--propagation", "private"]
        if not network:
            ns.append("--net")
        argv = ["sudo", "-n", "--"] + ns + ["/bin/sh", "-c", _NS_SCRIPT, "sic-ns", self.user,
                                         str(self.mem_mb * 1024 * 1024), str(self.nproc),
                                         "/usr/bin/python3", "-I", "-c", _BOOTSTRAP]
        try:
            p = subprocess.run(argv, input=job, capture_output=True, text=True, timeout=timeout + 20,
                               env={"PATH": SAFE_PATH + ":/usr/sbin:/sbin"})
            return _parse(p.stdout, p.stderr, p.returncode)
        except subprocess.TimeoutExpired:
            return RunResult(None, "", "", timed_out=True, error="outer timeout")


def default_sandbox(prefer_user: bool = True):
    if prefer_user and UserSandbox.available():
        return UserSandbox()
    return LocalSandbox()
