"""Lightning backend + deploy script logic, with mocks (no Lightning account / no spend)."""
import importlib.util
import os
import time

import pytest

from sic import llm as L
from sic.llm import FallbackLLM, HeartbeatLLM, MockLLM

from test_modal_backend import KEY, Mock, mock_url  # noqa: F401  (reuse the fake vLLM server)

HERE = os.path.dirname(__file__)


def _deploy():
    spec = importlib.util.spec_from_file_location("lightning_llm", os.path.join(HERE, "..", "deploy", "lightning_llm.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_lightning_backend_selected_with_heartbeat(monkeypatch, tmp_path):
    keyf = tmp_path / "k"; keyf.write_text("abc\n")
    monkeypatch.setenv("SIC_BACKEND", "lightning")
    monkeypatch.setenv("SIC_LIGHTNING_KEY_FILE", str(keyf))
    monkeypatch.setenv("SIC_LIGHTNING_URL", "https://8000-abc.cloudspaces.litng.ai")
    monkeypatch.setenv("SIC_LIGHTNING_HEARTBEAT", str(tmp_path / "hb"))
    assert L.available_models()[0] == "dolphin3-8b"
    m = L.make_llm("dolphin3-8b")
    assert isinstance(m, FallbackLLM) and isinstance(m.primary, HeartbeatLLM)
    assert m.primary.inner.base_url == "https://8000-abc.cloudspaces.litng.ai" and m.primary.inner.api_key == "abc"
    assert m.fallback.model == "dolphin3-3b"
    assert isinstance(L.make_llm("dolphin3-3b"), L.OllamaLLM)
    monkeypatch.setenv("SIC_LIGHTNING_KEY_FILE", str(tmp_path / "none"))
    assert isinstance(L.make_llm("dolphin3-8b"), L.OllamaLLM)  # no key -> local only


def test_heartbeat_and_streaming_through_lightning(mock_url, tmp_path):  # noqa: F811
    Mock.replies = ["hey Colby"]
    hb = tmp_path / "hb"
    inner = L.OpenAICompatLLM(mock_url, KEY, "dolphin3-8b")
    fb = FallbackLLM(HeartbeatLLM(inner, str(hb)), MockLLM(["local"]))
    toks = []
    assert fb.chat([{"role": "user", "content": "hi"}], on_token=toks.append) == "hey Colby"
    assert fb.last_backend == "modal" and abs(float(hb.read_text()) - time.time()) < 5 and toks


def test_down_studio_triggers_autostart_once_and_falls_back(tmp_path):
    marker = tmp_path / "started"
    cmd = ["bash", "-c", f"echo x >> {marker}"]
    inner = L.OpenAICompatLLM("http://127.0.0.1:9", KEY, "dolphin3-8b", first_token_timeout=1)
    fb = FallbackLLM(HeartbeatLLM(inner, str(tmp_path / "hb"), autostart_cmd=cmd), MockLLM(["a", "b"]), cooldown=0)
    assert fb.chat([{"role": "user", "content": "x"}]) == "a"
    assert fb.chat([{"role": "user", "content": "y"}]) == "b"
    time.sleep(0.5)
    assert marker.read_text().count("x") == 1  # rate-limited: started once


def test_watchdog_policy():
    d = _deploy()
    assert d.should_stop(11 * 60, 30 * 60, idle_min=10, max_up_min=120) == "idle"
    assert d.should_stop(60, 121 * 60, idle_min=10, max_up_min=120) == "max-uptime"
    assert d.should_stop(60, 30 * 60, idle_min=10, max_up_min=120) is None


def test_endpoint_url_parsing():
    d = _deploy()

    class EP:
        urls = ["http://insecure", "https://8000-01abc.cloudspaces.litng.ai/"]
    assert d.endpoint_url([EP()]) == "https://8000-01abc.cloudspaces.litng.ai"
    assert d.endpoint_url([]) is None


def test_up_and_down_with_fake_studio(monkeypatch, tmp_path):
    d = _deploy()
    for name in ("KEY_FILE", "URL_FILE", "LAST_USED", "STARTED"):
        monkeypatch.setattr(d, name, str(tmp_path / name.lower()))
    monkeypatch.setattr(d, "wait_healthy", lambda url, **k: 1.0)

    class EP:
        urls = ["https://8000-x.cloudspaces.litng.ai"]

    class FakeStudio:
        status = "Stopped"
        calls = []

        def start(self, machine, interruptible, max_runtime):
            self.calls.append(("start", str(machine), interruptible, max_runtime)); self.status = "Running"

        def set_env(self, env):
            self.calls.append(("env", sorted(env)))

        def run_and_detach(self, cmd, timeout):
            self.calls.append(("run", cmd))

        def list_ports(self):
            return []

        def add_ports(self, ports):
            self.calls.append(("ports", ports)); return [EP()]

        def stop(self):
            self.calls.append(("stop",)); self.status = "Stopped"
    fs = FakeStudio()
    url = d.up(fs)
    assert url == "https://8000-x.cloudspaces.litng.ai"
    assert open(d.URL_FILE).read().strip() == url
    assert oct(os.stat(d.KEY_FILE).st_mode)[-3:] == "600" and oct(os.stat(d.URL_FILE).st_mode)[-3:] == "600"
    kinds = [c[0] for c in fs.calls]
    assert kinds[:2] == ["start", "env"] and ("ports", {"vllm": 8000}) in fs.calls
    assert fs.calls[0][3] == d.MAX_UP_MIN * 60  # hard runtime cap passed to Lightning
    run = next(c[1] for c in fs.calls if c[0] == "run")
    assert "--api-key \"$VLLM_API_KEY\"" in run and "dolphin3-8b" in run and open(d.KEY_FILE).read().strip() not in run
    d.down(fs)
    assert fs.calls[-1] == ("stop",) and not os.path.exists(d.STARTED)


def test_missing_credentials_exit(monkeypatch):
    d = _deploy()
    monkeypatch.delenv("LIGHTNING_USER_ID", raising=False)
    monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        d.studio()


def test_url_file_is_reread(mock_url, tmp_path):  # noqa: F811
    Mock.replies = ["ok after start"]
    uf = tmp_path / "url"
    inner = L.OpenAICompatLLM("http://127.0.0.1:9", KEY, "dolphin3-8b", first_token_timeout=1)
    hb = HeartbeatLLM(inner, str(tmp_path / "hb"), url_file=str(uf))
    uf.write_text(mock_url + "\n")
    assert hb.chat([{"role": "user", "content": "x"}]) == "ok after start"
