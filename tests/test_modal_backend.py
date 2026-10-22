"""Modal / OpenAI-compatible backend, tested against a local mock vLLM server."""
import http.server
import json
import threading
import time

import pytest

from sic import llm as L
from sic.chat_agent import ChatAgent
from sic.llm import FallbackLLM, MockLLM, OpenAICompatLLM
from sic.sandbox import LocalSandbox
from sic.workspace import AgentComputer

KEY = "test-key-not-secret"


class Mock:
    replies: list = []
    seen: list = []
    delay = 0.0
    status = 200


def _handler():
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            Mock.seen.append((self.headers.get("Authorization"), body))
            time.sleep(Mock.delay)
            if self.headers.get("Authorization") != "Bearer " + KEY:
                self.send_response(401); self.end_headers(); return
            if Mock.status != 200:
                self.send_response(Mock.status); self.end_headers(); return
            text = Mock.replies.pop(0) if Mock.replies else "ok"
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for i in range(0, len(text), 4):
                    d = {"choices": [{"delta": {"content": text[i:i + 4]}}]}
                    self.wfile.write(f"data: {json.dumps(d)}\n\n".encode()); self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": text}}],
                                             "usage": {"completion_tokens": 3}}).encode())
    return H


@pytest.fixture
def mock_url():
    Mock.replies, Mock.seen, Mock.delay, Mock.status = [], [], 0.0, 200
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_streaming_and_auth_and_params(mock_url):
    Mock.replies = ["Hey Colby, what's up?"]
    llm = OpenAICompatLLM(mock_url, KEY, "dolphin3-8b")
    toks = []
    out = llm.chat([{"role": "user", "content": "hi"}], on_token=toks.append,
                   stop=["</tool>", "\nUser:", "\nAssistant:", "<tool_result", "extra"], num_predict=50)
    assert out == "Hey Colby, what's up?" and len(toks) > 1 and "".join(toks) == out
    auth, body = Mock.seen[0]
    assert auth == "Bearer " + KEY and body["model"] == "dolphin3-8b" and body["stream"]
    assert body["max_tokens"] == 50 and len(body["stop"]) == 4


def test_non_streaming(mock_url):
    Mock.replies = ["plain"]
    assert OpenAICompatLLM(mock_url, KEY, "m").chat([{"role": "user", "content": "x"}]) == "plain"


def test_wrong_key_falls_back(mock_url):
    fb = FallbackLLM(OpenAICompatLLM(mock_url, "wrong", "m"), MockLLM(["local answer"]))
    assert fb.chat([{"role": "user", "content": "x"}]) == "local answer" and fb.last_backend == "ollama"


def test_cold_start_timeout_falls_back_then_cooldown(mock_url):
    Mock.delay = 1.5
    fb = FallbackLLM(OpenAICompatLLM(mock_url, KEY, "m", first_token_timeout=0.5),
                     MockLLM(["local 1", "local 2"]), cooldown=60)
    toks = []
    t = time.time()
    assert fb.chat([{"role": "user", "content": "x"}], on_token=toks.append) == "local 1"
    assert time.time() - t < 1.4 and toks == ["local 1"]
    n = len(Mock.seen)
    assert fb.chat([{"role": "user", "content": "y"}]) == "local 2"  # in cooldown: modal skipped
    assert len(Mock.seen) == n


def test_unreachable_endpoint_falls_back():
    fb = FallbackLLM(OpenAICompatLLM("http://127.0.0.1:9", KEY, "m", first_token_timeout=1), MockLLM(["local"]))
    assert fb.chat([{"role": "user", "content": "x"}]) == "local"


def test_modal_used_when_healthy(mock_url):
    Mock.replies = ["from modal"]
    fb = FallbackLLM(OpenAICompatLLM(mock_url, KEY, "m"), MockLLM(["local"]))
    assert fb.chat([{"role": "user", "content": "x"}], on_token=lambda t: None) == "from modal"
    assert fb.last_backend == "modal"


def test_tool_calling_through_modal_backend(mock_url, tmp_path):
    Mock.replies = ['<tool>{"name": "write_file", "args": {"path": "a.txt", "content": "hi"}}', "Done, Colby!"]
    comp = AgentComputer(LocalSandbox(), root=str(tmp_path / "ws"))
    llm = FallbackLLM(OpenAICompatLLM(mock_url, KEY, "dolphin3-8b"), MockLLM([]))
    events = []
    out = ChatAgent(llm, LocalSandbox(), computer=comp, web=False).respond("create file a.txt", emit=events.append)
    assert (tmp_path / "ws" / "a.txt").read_text() == "hi" and out["answer"] == "Done, Colby!"
    assert "</tool>" in Mock.seen[0][1]["stop"]  # stop sequence forwarded


def test_make_llm_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("SIC_BACKEND", "ollama")
    assert isinstance(L.make_llm("dolphin3-3b"), L.OllamaLLM)
    monkeypatch.setenv("SIC_BACKEND", "modal")
    monkeypatch.setenv("SIC_MODAL_URL", "https://x--sic-llm-chat.modal.run")
    keyf = tmp_path / "k"; keyf.write_text("abc\n")
    monkeypatch.setenv("SIC_MODAL_API_KEY_FILE", str(keyf))
    monkeypatch.delenv("SIC_MODAL_API_KEY", raising=False)
    m = L.make_llm("dolphin3-8b")
    assert isinstance(m, FallbackLLM) and m.primary.api_key == "abc" and m.fallback.model == "dolphin3-3b"
    assert isinstance(L.make_llm("dolphin3-3b"), L.OllamaLLM)  # local models stay local
    assert L.available_models()[0] == "dolphin3-8b"
    monkeypatch.setenv("SIC_MODAL_API_KEY_FILE", str(tmp_path / "missing"))
    assert isinstance(L.make_llm("dolphin3-8b"), L.OllamaLLM)  # no key -> local only


def test_modal_app_definition_imports():
    pytest.importorskip("modal")
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("modal_llm", os.path.join(os.path.dirname(__file__), "..", "deploy", "modal_llm.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert "chat" in m.app.registered_functions and m.SCALEDOWN == 300 and m.GPU == "L4"
