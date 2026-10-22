"""Run Dolphin's model on a Lightning AI Studio GPU (vLLM, OpenAI-compatible, bearer key) from this box.

    python3 deploy/lightning_llm.py up        # start Studio on GPU, start vLLM, expose port, save URL
    python3 deploy/lightning_llm.py down      # stop the Studio now (stops credit burn)
    python3 deploy/lightning_llm.py status
    python3 deploy/lightning_llm.py watch     # idle watchdog: stop after SIC_LIGHTNING_IDLE_MIN idle

Auth: the SDK reads LIGHTNING_USER_ID and LIGHTNING_API_KEY from the environment (never printed).
Credit safety, three layers:
  1. `watch` (run under nohup on this box) stops the Studio after N minutes without chat traffic
     (sic touches state/llm_last_used on every request) or after SIC_LIGHTNING_MAX_UP_MIN of uptime.
  2. Lightning's own Studio auto-sleep is enabled with the same idle time.
  3. start(max_runtime=...) asks Lightning to release the machine after a hard cap.
Model: Dolphin3.0-Llama3.1-8B with on-the-fly FP8 on an L4 (24 GB; T4 has no FP8/bf16 and 16 GB
is too small for the fp16 weights). The vLLM bearer key comes from gitignored .lightning_llm_key.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO = os.environ.get("SIC_LIGHTNING_STUDIO", "sic-llm")
TEAMSPACE = os.environ.get("SIC_LIGHTNING_TEAMSPACE") or None  # default: the user's default teamspace
MACHINE = os.environ.get("SIC_LIGHTNING_MACHINE", "L4")
INTERRUPTIBLE = os.environ.get("SIC_LIGHTNING_INTERRUPTIBLE", "0") == "1"
IDLE_MIN = float(os.environ.get("SIC_LIGHTNING_IDLE_MIN", "10"))
MAX_UP_MIN = float(os.environ.get("SIC_LIGHTNING_MAX_UP_MIN", "120"))
REPO = os.environ.get("SIC_LIGHTNING_MODEL", "dphn/Dolphin3.0-Llama3.1-8B")
SERVED = "dolphin3-8b"
PORT = 8000
KEY_FILE = os.path.join(ROOT, ".lightning_llm_key")
URL_FILE = os.path.join(ROOT, ".lightning_url")
LAST_USED = os.path.join(ROOT, "state", "llm_last_used")
STARTED = os.path.join(ROOT, "state", "lightning_started")

VLLM_CMD = (
    "mkdir -p ~/sic && cd ~/sic && "
    "(python -c 'import vllm' 2>/dev/null || pip install -q 'vllm>=0.11' > pip.log 2>&1) && "
    "pkill -f 'vllm serve' ; "
    f"nohup vllm serve {REPO} --served-model-name {SERVED} --host 0.0.0.0 --port {PORT} "
    "--quantization fp8 --max-model-len 16384 --gpu-memory-utilization 0.92 --enable-prefix-caching "
    "--api-key \"$VLLM_API_KEY\" > vllm.log 2>&1 &"
)


def _key() -> str:
    if not os.path.exists(KEY_FILE):
        fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32) + "\n")
    os.chmod(KEY_FILE, 0o600)
    return open(KEY_FILE).read().strip()


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text + "\n")
    os.replace(path + ".tmp", path)


def _require_auth() -> None:
    missing = [k for k in ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit("missing env: " + ", ".join(missing))


def studio():
    _require_auth()
    from lightning_sdk import Studio
    return Studio(STUDIO, teamspace=TEAMSPACE, create_ok=True)


def endpoint_url(endpoints) -> str | None:
    for ep in endpoints or []:
        for u in getattr(ep, "urls", None) or []:
            if u.startswith("https://"):
                return u.rstrip("/")
    return None


def wait_healthy(url: str, timeout: float = 1200, interval: float = 10) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/health", timeout=10) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError("vLLM did not become healthy")


def up(s=None) -> str:
    from lightning_sdk import Machine
    s = s or studio()
    key = _key()
    st = str(s.status)
    if "Running" not in st:
        print(f"starting studio {STUDIO} on {MACHINE} (interruptible={INTERRUPTIBLE})...", flush=True)
        s.start(machine=getattr(Machine, MACHINE), interruptible=INTERRUPTIBLE,
                max_runtime=int(MAX_UP_MIN * 60))
    try:
        s.auto_sleep = True
        s.auto_sleep_time = int(IDLE_MIN * 60)
    except Exception as e:  # not fatal: our watchdog still stops it
        print("warning: could not set auto-sleep:", type(e).__name__, flush=True)
    s.set_env({"VLLM_API_KEY": key, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    s.run_and_detach(VLLM_CMD, timeout=30)
    url = endpoint_url(s.list_ports()) or endpoint_url(s.add_ports({"vllm": PORT}))
    if not url:
        raise RuntimeError("no public URL returned for port %d" % PORT)
    _write(URL_FILE, url)
    _write(STARTED, str(time.time()))
    touch()
    print("endpoint saved to .lightning_url:", url, flush=True)
    secs = wait_healthy(url)
    print(f"vLLM healthy after {secs:.0f}s", flush=True)
    return url


def down(s=None) -> None:
    s = s or studio()
    if any(x in str(s.status) for x in ("Running", "Pending")):
        s.stop()
        print("studio stopped", flush=True)
    else:
        print("studio not running:", s.status, flush=True)
    try:
        os.remove(STARTED)
    except OSError:
        pass


def touch() -> None:
    _write(LAST_USED, str(time.time()))


def _age(path: str) -> float:
    try:
        return time.time() - float(open(path).read().strip())
    except (OSError, ValueError):
        return float("inf")


def should_stop(now_idle_s: float, up_s: float, idle_min: float = IDLE_MIN, max_up_min: float = MAX_UP_MIN):
    if now_idle_s >= idle_min * 60:
        return "idle"
    if up_s >= max_up_min * 60:
        return "max-uptime"
    return None


def watch(poll: float = 30, s_factory=None) -> None:
    s_factory = s_factory or studio
    while True:
        try:
            if os.path.exists(STARTED):
                why = should_stop(_age(LAST_USED), _age(STARTED))
                if why:
                    print(time.strftime("%FT%T"), "stopping studio:", why, flush=True)
                    down(s_factory())
        except Exception as e:
            print(time.strftime("%FT%T"), "watchdog error:", type(e).__name__, str(e)[:200], flush=True)
        time.sleep(poll)


def status() -> None:
    s = studio()
    print(json.dumps({"studio": STUDIO, "status": str(s.status), "url_file": URL_FILE,
                      "idle_s": round(_age(LAST_USED)), "up_s": round(_age(STARTED))}))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"up": up, "down": down, "status": status, "watch": watch}[cmd]()
