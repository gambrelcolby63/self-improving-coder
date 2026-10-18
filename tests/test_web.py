import socket

import pytest

from sic import web
from sic.web import BlockedURL, check_url, fetch_url, parse_ddg

BLOCKED = ["http://127.0.0.1:11434/api/tags", "http://localhost:8765", "http://LOCALHOST.",
           "http://foo.localhost/", "http://[::1]:11434/", "http://0.0.0.0/", "http://2130706433/",
           "http://0x7f000001/", "http://0177.0.0.1/", "http://127.1/", "http://10.0.0.2/",
           "http://172.17.0.1:2375/", "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data/",
           "http://metadata.google.internal/", "http://100.64.0.1/", "http://[fd00::1]/",
           "http://[::ffff:127.0.0.1]/", "http://printer.local/", "http://intranet/",
           "file:///etc/passwd", "ftp://example.com/", "gopher://x.org/", "http://198.18.0.1/"]


@pytest.mark.parametrize("url", BLOCKED)
def test_blocklist(url):
    with pytest.raises(BlockedURL):
        check_url(url, resolve=False)


def _fake_dns(ip):
    def f(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
    return f


def test_public_host_allowed(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns("93.184.216.34"))
    assert check_url("https://example.com/x")


def test_dns_to_private_is_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns("127.0.0.1"))
    with pytest.raises(BlockedURL):
        check_url("https://evil-rebind.example/")
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns("169.254.169.254"))
    with pytest.raises(BlockedURL):
        check_url("https://evil.example/")


def test_egress_proxy_range_only_via_dns(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns("198.18.0.1"))
    assert check_url("https://pypi.org/")
    monkeypatch.setenv("SIC_ALLOW_RESOLVED_NETS", "")
    with pytest.raises(BlockedURL):
        check_url("https://pypi.org/")


def test_fetch_url_returns_error_not_exception():
    r = fetch_url("http://127.0.0.1:11434/api/tags")
    assert r["error"].startswith("blocked")


def test_redirect_to_private_is_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns("93.184.216.34"))
    h = web._SafeRedirect()

    class Req:
        full_url = "https://example.com/a"
    with pytest.raises(BlockedURL):
        h.redirect_request(Req(), None, 302, "Found", {}, "http://169.254.169.254/latest")


def test_html_to_text_strips_scripts():
    title, text = web.html_to_text("<html><head><title>T</title><script>evil()</script></head>"
                                   "<body><p>Hello</p><style>x{}</style><p>World</p></body></html>")
    assert title == "T" and "evil" not in text and "Hello" in text and "World" in text


def test_parse_ddg():
    page = ('<div><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=x">'
            'Python <b>docs</b></a></div><a class="result__snippet" href="x">The <b>official</b> docs</a>'
            '<a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a>')
    r = parse_ddg(page)
    assert r == [{"title": "Python docs", "url": "https://docs.python.org/3/", "snippet": "The official docs"}]
