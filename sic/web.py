"""web_search (DuckDuckGo HTML endpoint, no API key) and fetch_url (text extraction).

SSRF guard: only http(s); hostnames like localhost / *.local / *.internal / metadata
are refused; IP literals (incl. decimal/hex/octal IPv4 forms) and every resolved
address must be public.  Redirects are re-validated hop by hop.  Size and time are
capped.  Everything returned is UNTRUSTED data: callers wrap it in <untrusted> tags
and the model is told never to follow instructions found inside.

Note: some sandboxes (like the box this was built on) use a fake-IP egress proxy
that resolves every public hostname to 198.18.0.0/15.  Addresses in
SIC_ALLOW_RESOLVED_NETS (default "198.18.0.0/15") are therefore accepted when they
come from DNS resolution, but never as literals typed into a URL.
"""
from __future__ import annotations

import html
import ipaddress
import os
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = "Mozilla/5.0 (X11; Linux x86_64) self-improving-coder/0.1 (+https://github.com/gambrelcolby63/self-improving-coder)"
BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".home.arpa", ".lan")
BLOCKED_HOSTS = {"localhost", "metadata", "metadata.google.internal", "instance-data", "ip6-localhost"}
EXTRA_BLOCKED = [ipaddress.ip_network(n) for n in ("100.64.0.0/10", "169.254.0.0/16", "0.0.0.0/8",
                                                   "fd00::/8", "fe80::/10")]


def _allowed_resolved():
    raw = os.environ.get("SIC_ALLOW_RESOLVED_NETS", "198.18.0.0/15")
    return [ipaddress.ip_network(n.strip()) for n in raw.split(",") if n.strip()]


class BlockedURL(ValueError):
    pass


def _parse_ip_literal(host: str):
    h = host.strip("[]")
    try:
        return ipaddress.ip_address(h)
    except ValueError:
        pass
    # inet_aton accepts legacy forms: 2130706433, 0x7f000001, 0177.0.0.1, 127.1
    if re.fullmatch(r"[0-9a-fA-FxX.]+", h) and not re.fullmatch(r"[a-fA-F.]+", h):
        try:
            return ipaddress.ip_address(socket.inet_aton(h))
        except OSError:
            pass
    return None


def ip_is_public(ip) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
            or ip.is_unspecified):
        return False
    return not any(ip in n for n in EXTRA_BLOCKED if n.version == ip.version)


def check_url(url: str, resolve: bool = True) -> str:
    """Raise BlockedURL if the URL may target a local/private service. Returns the URL."""
    p = urllib.parse.urlsplit(url)
    if p.scheme not in ("http", "https"):
        raise BlockedURL(f"scheme {p.scheme!r} not allowed")
    host = (p.hostname or "").lower().rstrip(".")
    if not host:
        raise BlockedURL("missing host")
    if host in BLOCKED_HOSTS or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise BlockedURL(f"host {host!r} is local")
    lit = _parse_ip_literal(host)
    if lit is not None:
        if not ip_is_public(lit):
            raise BlockedURL(f"address {lit} is not public")
        return url
    if "." not in host:
        raise BlockedURL(f"single-label host {host!r} not allowed")
    if resolve:
        try:
            infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80),
                                       proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            raise BlockedURL(f"cannot resolve {host}: {e}")
        allow = _allowed_resolved()
        for info in infos:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if any(ip in n for n in allow if n.version == ip.version):
                continue
            if not ip_is_public(ip):
                raise BlockedURL(f"{host} resolves to non-public {ip}")
    return url


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirect)


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.depth, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "pre", "tr"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self.depth:
            self.out.append(data)


def html_to_text(raw: str) -> tuple[str, str]:
    p = _Text()
    p.feed(raw)
    text = re.sub(r"[ \t]+", " ", "".join(p.out))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return p.title.strip(), text


def _get(url: str, data: bytes | None = None, timeout: float = 10, max_bytes: int = 2_000_000):
    check_url(url)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept-Language": "en"})
    with _opener.open(req, timeout=timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        body = r.read(max_bytes + 1)
        return r.geturl(), ctype, body[:max_bytes], len(body) > max_bytes


def fetch_url(url: str, max_chars: int = 8000, timeout: float = 10) -> dict:
    try:
        final, ctype, body, truncated = _get(url, timeout=timeout)
    except BlockedURL as e:
        return {"url": url, "error": f"blocked: {e}"}
    except Exception as e:  # network errors are data, not crashes
        return {"url": url, "error": f"{type(e).__name__}: {e}"}
    if not any(t in ctype for t in ("text/", "json", "xml", "javascript")) and ctype:
        return {"url": final, "error": f"unsupported content-type {ctype}"}
    raw = body.decode("utf-8", "replace")
    title, text = html_to_text(raw) if "html" in ctype or "<html" in raw[:500].lower() else ("", raw)
    return {"url": final, "title": title, "text": text[:max_chars],
            "truncated": truncated or len(text) > max_chars}


def web_search(query: str, k: int = 5, timeout: float = 10) -> list[dict]:
    data = urllib.parse.urlencode({"q": query}).encode()
    try:
        _, _, body, _ = _get("https://html.duckduckgo.com/html/", data=data, timeout=timeout)
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]
    return parse_ddg(body.decode("utf-8", "replace"), k)


def parse_ddg(page: str, k: int = 5) -> list[dict]:
    strip = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()
    out = []
    for chunk in page.split('class="result__a"')[1:]:
        m = re.search(r'href="([^"]+)"[^>]*>(.*?)</a>', chunk, re.S)
        if not m:
            continue
        href = html.unescape(m.group(1))
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
        if "uddg" in q:
            href = q["uddg"][0]
        if href.startswith("//"):
            href = "https:" + href
        if "duckduckgo.com/y.js" in href:  # ads
            continue
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', chunk, re.S)
        out.append({"title": strip(m.group(2)), "url": href, "snippet": strip(sn.group(1) if sn else "")})
        if len(out) >= k:
            break
    return out
