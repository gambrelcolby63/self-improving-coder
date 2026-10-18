import os

import pytest

from sic.sandbox import LocalSandbox, UserSandbox
from sic.workspace import AgentComputer, PathError, safe_relpath


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x", "a/../../x", "~/.ssh/id_rsa", "..", "a\x00b",
                                 "/home/box/.config/gh/hosts.yml"])
def test_safe_relpath_rejects(bad):
    with pytest.raises(PathError):
        safe_relpath(bad)


@pytest.mark.parametrize("good,norm", [("a.py", "a.py"), ("./src/../a.py", "a.py"), ("", "."), ("d/e/f.txt", "d/e/f.txt")])
def test_safe_relpath_accepts(good, norm):
    assert safe_relpath(good) == norm


@pytest.fixture
def local_computer(tmp_path):
    return AgentComputer(LocalSandbox(), root=str(tmp_path / "ws"))


def test_files_persist_and_tests_run(local_computer, tmp_path):
    c = local_computer
    assert c.write_file("pkg/m.py", "def f():\n    return 42\n")["written"] == "pkg/m.py"
    c.write_file("pkg/test_m.py", "import unittest\nfrom m import f\nclass T(unittest.TestCase):\n"
                 "    def test_f(self):\n        self.assertEqual(f(), 42)\n")
    assert c.read_file("pkg/m.py")["content"].startswith("def f")
    names = [e["name"] for e in c.list_dir("pkg")["entries"]]
    assert "m.py" in names
    assert c.run_tests("pkg")["passed"]
    # a new AgentComputer on the same root sees the same files (persistence across chats)
    c2 = AgentComputer(LocalSandbox(), root=c.root)
    assert "return 42" in c2.read_file("pkg/m.py")["content"]


def test_tool_call_denies_escape(local_computer):
    assert "denied" in local_computer.call("read_file", {"path": "../../etc/passwd"})["error"]
    assert "denied" in local_computer.call("write_file", {"path": "/tmp/x", "content": "x"})["error"]
    assert "unknown tool" in local_computer.call("rm_rf", {})["error"]


def test_symlink_escape_blocked(local_computer, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    os.symlink(str(secret), os.path.join(local_computer.root, "link"))
    r = local_computer.read_file("link")
    assert "TOPSECRET" not in str(r) and "escapes" in r.get("error", "")


def test_command_env_is_scrubbed(local_computer, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake_for_test")
    r = local_computer.run_command("env")
    assert "ghp_fake_for_test" not in r["stdout"] and "GITHUB_TOKEN" not in r["stdout"]


# ---- real confinement (only where the sicagent user + sudo exist, i.e. the dev box) ----
needs_user = pytest.mark.skipif(not UserSandbox.available(), reason="sandbox user not configured")


@pytest.fixture
def user_computer():
    return AgentComputer(UserSandbox())


@needs_user
def test_user_sandbox_identity_and_no_sudo(user_computer):
    r = user_computer.run_command("id -un; sudo -n true && echo SUDO_WORKS || echo NO_SUDO")
    assert r["stdout"].split()[0] == "sicagent" and "NO_SUDO" in r["stdout"]


@needs_user
def test_user_sandbox_hides_host_files(user_computer):
    r = user_computer.run_command(
        "ls -A /workspace | wc -l; cat /home/box/.config/gh/hosts.yml 2>&1 | head -c 80; "
        "ls /home; ls /cursor 2>/dev/null | wc -l; ps -e --no-headers | wc -l")
    out = r["stdout"].split("\n")
    assert out[0] == "0"  # /workspace (other repos, pump, Dolphin...) is an empty tmpfs
    assert "oauth_token" not in r["stdout"] and "github.com" not in r["stdout"]
    assert int(out[-2]) < 10  # private PID namespace: only our own processes


@needs_user
def test_user_sandbox_env_has_no_tokens(user_computer, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake_for_test")
    r = user_computer.run_command("env")
    assert "ghp_fake" not in r["stdout"] and "TOKEN" not in r["stdout"]


@needs_user
@pytest.mark.parametrize("target", ["127.0.0.1:11434", "172.17.0.1:2375", "169.254.169.254:80"])
def test_user_sandbox_cannot_reach_local_services(user_computer, target):
    r = user_computer.run_command(f"curl -s -m 4 -o /dev/null -w '%{{http_code}}' http://{target}/ || true")
    assert r["stdout"].strip() in ("000", "")


@needs_user
def test_eval_runs_have_no_network():
    r = UserSandbox().run(["python3", "-c", "import socket;socket.create_connection(('1.1.1.1',53),2)"],
                          timeout=5, network=False)
    assert not r.ok
