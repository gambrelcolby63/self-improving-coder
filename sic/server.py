"""Single-user, password-protected, mobile-friendly chat UI (stdlib only).

    python -m sic.server            # binds 127.0.0.1:8765

* Password: read from $SIC_PASSWORD_FILE (default ~/.config/sic/chat_password). If the
  file does not exist a random one is generated and saved with mode 600. It is never
  printed or logged.
* Sessions: random token in an HttpOnly, SameSite=Strict cookie, held only in memory.
* No chat history is stored server-side: the browser keeps the conversation in memory
  (lost on reload). "Learn from verified answers" is an opt-in toggle.
* Binds to localhost only. Exposing it is left to the operator (see README).
"""
from __future__ import annotations

import hmac
import http.server
import json
import os
import secrets
import socketserver
import stat
import sys
import threading
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PW_FILE = os.environ.get("SIC_PASSWORD_FILE", os.path.expanduser("~/.config/sic/chat_password"))
SESSION_TTL = 12 * 3600


def load_or_create_password(path: str = PW_FILE) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(18) + "\n")
    st = os.stat(path)
    if stat.S_IMODE(st.st_mode) & 0o077:
        os.chmod(path, 0o600)
    with open(path) as f:
        return f.read().strip()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>SIC coder</title><style>
:root{--bg:#0f1115;--fg:#e6e6e6;--mut:#8b93a1;--card:#1a1d24;--acc:#3b82f6;--ok:#16a34a;--bad:#dc2626}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,sans-serif;display:flex;flex-direction:column;height:100dvh}
header{padding:10px 14px;border-bottom:1px solid #262a33;display:flex;justify-content:space-between;align-items:center}
header b{font-size:16px}header label{font-size:12px;color:var(--mut)}
#log{flex:1;overflow-y:auto;padding:12px}
.msg{margin:8px 0;padding:10px 12px;border-radius:12px;max-width:100%;white-space:pre-wrap;word-wrap:break-word}
.user{background:var(--acc);color:#fff;margin-left:15%}.bot{background:var(--card)}
.ev{font:12px ui-monospace,monospace;color:var(--mut);margin:3px 0 3px 4px;white-space:pre-wrap}
pre{background:#0b0d11;padding:10px;border-radius:8px;overflow-x:auto;font:13px ui-monospace,monospace;white-space:pre}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600;margin-top:6px}
.pass{background:var(--ok)}.fail{background:var(--bad)}
form{display:flex;gap:8px;padding:10px;border-top:1px solid #262a33;padding-bottom:calc(10px + env(safe-area-inset-bottom))}
textarea{flex:1;resize:none;height:46px;border-radius:10px;border:1px solid #333;background:var(--card);color:var(--fg);padding:10px;font:15px system-ui}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:0 16px;font-weight:600}
a{color:#7fb0ff}details summary{cursor:pointer;color:var(--mut);font-size:12px}
</style></head><body>
<header><b>SIC coder</b><label><input type="checkbox" id="learn"> learn from verified answers</label></header>
<div id="log"><div class="ev">Local coding agent. Code is run in a sandbox and marked PASS/FAIL. Chat lives only in this tab.</div></div>
<form id="f"><textarea id="q" placeholder="Ask a coding question..."></textarea><button>Send</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q');let hist=[];
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function md(s){let out='',i=0;const re=/```(\w*)\n([\s\S]*?)```/g;let m;while((m=re.exec(s))){out+=esc(s.slice(i,m.index))+'<pre>'+esc(m[2])+'</pre>';i=re.lastIndex}return out+esc(s.slice(i))}
function add(cls,html){const d=document.createElement('div');d.className=cls;d.innerHTML=html;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
function evText(e){if(e.type==='tool')return '🔧 '+e.name+'('+esc(JSON.stringify(e.args)).slice(0,200)+')'+(e.auto?' [auto]':'');
 if(e.type==='tool_result')return '   ↳ '+esc(e.preview.replace(/<\/?untrusted[^>]*>/g,'')).slice(0,300);
 if(e.type==='sandbox')return '🧪 sandbox run #'+e.attempt+' ('+e.lang+'): '+(e.passed?'PASS':'FAIL');
 if(e.type==='experience')return '📚 recalled past solution '+esc(e.id||'');
 if(e.type==='docs')return '📄 docs: '+esc(e.sources.join(', '));
 if(e.type==='learned')return '🧠 saved to experience library';if(e.type==='error')return '⚠️ '+esc(e.message);return ''}
document.getElementById('f').onsubmit=async ev=>{ev.preventDefault();const text=q.value.trim();if(!text)return;q.value='';
 add('msg user',esc(text));const evbox=add('ev','thinking…');
 const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,history:hist,learn:document.getElementById('learn').checked})});
 if(r.status===401){location.href='/login';return}
 const rd=r.body.getReader(),dec=new TextDecoder();let buf='',lines=[];
 while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let k;
  while((k=buf.indexOf('\n'))>=0){const line=buf.slice(0,k);buf=buf.slice(k+1);if(!line)continue;const e=JSON.parse(line);
   if(e.type==='final'){let h=md(e.answer);if(e.verdict){h+='<br><span class="badge '+(e.verdict.passed?'pass':'fail')+'">sandbox '+(e.verdict.passed?'PASS':'FAIL')+' ('+e.verdict.lang+', attempt '+e.verdict.attempt+')</span><details><summary>run log</summary><pre>'+esc(e.verdict.log)+'</pre></details>'}
    if(e.sources&&e.sources.length){h+='<div class="ev">'+e.sources.map((s,i)=>'['+(i+1)+'] <a href="'+encodeURI(s.url)+'" target="_blank" rel="noopener noreferrer">'+esc(s.title||s.url)+'</a>').join('<br>')+'</div>'}
    h+='<div class="ev">'+e.seconds+'s</div>';add('msg bot',h);hist.push({role:'user',content:text},{role:'assistant',content:e.answer})}
   else{const t=evText(e);if(t){lines.push(t);evbox.textContent=lines.join('\n')}}}}
};
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&window.innerWidth>700){e.preventDefault();document.getElementById('f').requestSubmit()}});
</script></body></html>"""

LOGIN = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SIC login</title><style>body{background:#0f1115;color:#eee;font:16px system-ui;display:flex;height:100vh;align-items:center;justify-content:center}
form{display:flex;flex-direction:column;gap:10px;width:260px}input,button{padding:12px;border-radius:10px;border:1px solid #333;font-size:16px}
button{background:#3b82f6;color:#fff;border:0}</style></head><body><form method="post" action="/login">
<b>SIC coder</b><input type="password" name="password" placeholder="password" autofocus>
<button>Log in</button><small>%MSG%</small></form></body></html>"""


class App:
    def __init__(self, agent_factory, password: str):
        self.agent_factory = agent_factory
        self._agent = None
        self.password = password
        self.sessions: dict[str, float] = {}
        self.failures: list[float] = []
        self.lock = threading.Lock()

    @property
    def agent(self):
        if self._agent is None:
            self._agent = self.agent_factory()
        return self._agent

    def check_password(self, pw: str) -> bool:
        now = time.time()
        self.failures = [t for t in self.failures if now - t < 300]
        if len(self.failures) >= 10:
            return False  # crude lockout: 10 failures / 5 min
        ok = hmac.compare_digest(pw.strip().encode(), self.password.encode())
        if not ok:
            self.failures.append(now)
        return ok

    def new_session(self) -> str:
        tok = secrets.token_urlsafe(32)
        self.sessions[tok] = time.time() + SESSION_TTL
        return tok

    def valid(self, tok: str | None) -> bool:
        exp = self.sessions.get(tok or "")
        return bool(exp and exp > time.time())


def make_handler(app: App):
    class H(http.server.BaseHTTPRequestHandler):
        server_version = "sic"
        sys_version = ""

        def log_message(self, fmt, *args):  # never log bodies / passwords
            sys.stderr.write("%s %s\n" % (self.command, self.path.split("?")[0]))

        def _cookie(self):
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == "sic_session":
                    return v
            return None

        def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/healthz":
                return self._send(200, "ok", "text/plain")
            if path == "/login":
                return self._send(200, LOGIN.replace("%MSG%", ""))
            if not app.valid(self._cookie()):
                return self._send(303, "", extra={"Location": "/login"})
            if path == "/":
                return self._send(200, PAGE)
            self._send(404, "not found", "text/plain")

        def _body(self, limit=200_000):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(min(n, limit))

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/login":
                form = urllib.parse.parse_qs(self._body(10_000).decode(errors="replace"))
                if app.check_password((form.get("password") or [""])[0]):
                    secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
                    tok = app.new_session()
                    return self._send(303, "", extra={
                        "Location": "/",
                        "Set-Cookie": f"sic_session={tok}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_TTL}{secure}"})
                time.sleep(1)
                return self._send(401, LOGIN.replace("%MSG%", "wrong password"))
            if path == "/logout":
                app.sessions.pop(self._cookie() or "", None)
                return self._send(303, "", extra={"Location": "/login"})
            if path != "/api/chat":
                return self._send(404, "not found", "text/plain")
            if not app.valid(self._cookie()):
                return self._send(401, json.dumps({"error": "auth"}), "application/json")
            try:
                req = json.loads(self._body() or b"{}")
                msg = str(req.get("message", ""))[:8000]
            except json.JSONDecodeError:
                return self._send(400, "bad json", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True

            def emit(e):
                try:
                    self.wfile.write((json.dumps(e) + "\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            with app.lock:  # single user, one generation at a time
                try:
                    app.agent.respond(msg, req.get("history") or [], emit=emit, learn=bool(req.get("learn")))
                except Exception as ex:  # report, don't crash the server
                    emit({"type": "error", "message": f"{type(ex).__name__}: {ex}"[:500]})
                    emit({"type": "final", "answer": "Sorry, something went wrong.", "verdict": None,
                          "sources": [], "seconds": 0})

    return H


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def default_agent():
    from .chat_agent import ChatAgent
    from .experience import ExperienceLibrary
    from .ingest import load_store
    from .llm import OllamaLLM
    from .sandbox import default_sandbox
    from .workspace import AgentComputer
    sb = default_sandbox()
    lib = ExperienceLibrary(os.environ.get("SIC_CHAT_LIBRARY", os.path.join(ROOT, "store", "experience_chat.jsonl")))
    # seed retrieval with the verified training-split solutions
    train = os.path.join(ROOT, "store", "experience_train.jsonl")
    if os.path.exists(train):
        for it in ExperienceLibrary(train).items:
            if it["id"] not in {x["id"] for x in lib.items}:
                lib.items.append(it)
        lib._reindex()
    return ChatAgent(OllamaLLM(num_predict=1024), sb, AgentComputer(sb), docs=load_store(), experience=lib)


def main():
    host = os.environ.get("SIC_HOST", "127.0.0.1")
    port = int(os.environ.get("SIC_PORT", "8765"))
    app = App(default_agent, load_or_create_password())
    srv = Server((host, port), make_handler(app))
    print(f"SIC chat on http://{host}:{port}  (password file: {PW_FILE})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
