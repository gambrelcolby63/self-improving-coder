"""Single-user, password-protected, mobile-friendly chat UI (stdlib only).

    python -m sic.server            # binds 127.0.0.1:8765

* Password: read from $SIC_PASSWORD_FILE (default <repo>/.password, gitignored). If the
  file does not exist a random one is generated and saved with mode 600. It is never
  printed or logged.
* Sessions: random token in an HttpOnly, SameSite=Strict cookie, held only in memory.
* Conversations are stored server-side as JSON in $SIC_CONV_DIR (default <repo>/state/conversations,
  gitignored) so they survive reloads/restarts. "Learn from verified answers" is opt-in.
* Model picker: $SIC_MODELS (comma list); first is the default.
* Binds to localhost only. Exposing it is left to the operator (see README).
"""
from __future__ import annotations

import hmac
import http.server
import json
import os
import re
import secrets
import socketserver
import stat
import sys
import threading
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PW_FILE = os.environ.get("SIC_PASSWORD_FILE", os.path.join(ROOT, ".password"))
CONV_DIR = os.environ.get("SIC_CONV_DIR", os.path.join(ROOT, "state", "conversations"))
MODELS = [m for m in os.environ.get("SIC_MODELS", "dolphin3-3b,sic-qwen2.5-coder-1.5b").split(",") if m]
CONV_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,40}$")


class ConversationStore:
    """One JSON file per conversation: {id, title, model, updated, messages:[{role, content, ...}]}."""

    def __init__(self, root: str = CONV_DIR):
        self.root = root
        os.makedirs(root, mode=0o700, exist_ok=True)

    def _path(self, cid: str) -> str:
        if not CONV_ID_RE.match(cid or ""):
            raise ValueError("bad conversation id")
        return os.path.join(self.root, cid + ".json")

    def new(self, model: str) -> dict:
        return {"id": secrets.token_urlsafe(9), "title": "New chat", "model": model,
                "updated": time.time(), "messages": []}

    def get(self, cid: str) -> dict | None:
        try:
            with open(self._path(cid)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def save(self, conv: dict) -> None:
        conv["updated"] = time.time()
        p = self._path(conv["id"])
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(conv, f)
        os.replace(tmp, p)

    def delete(self, cid: str) -> None:
        try:
            os.remove(self._path(cid))
        except (OSError, ValueError):
            pass

    def list(self) -> list[dict]:
        out = []
        for n in os.listdir(self.root):
            if n.endswith(".json"):
                c = self.get(n[:-5])
                if c:
                    out.append({k: c[k] for k in ("id", "title", "model", "updated")})
        return sorted(out, key=lambda c: -c["updated"])
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
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="theme-color" content="#0f1115">
<title>Dolphin</title><style>
:root{--bg:#0f1115;--fg:#e6e6e6;--mut:#8b93a1;--card:#1a1d24;--acc:#3b82f6;--ok:#16a34a;--bad:#dc2626}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,sans-serif;display:flex;flex-direction:column;height:100dvh}
header{padding:8px 10px;border-bottom:1px solid #262a33;display:flex;gap:8px;align-items:center}
header b{font-size:16px;flex:1}select,.hb{background:var(--card);color:var(--fg);border:1px solid #333;border-radius:8px;padding:6px 8px;font-size:14px}
#side{position:fixed;inset:0 25% 0 0;max-width:340px;background:#14161b;border-right:1px solid #262a33;transform:translateX(-105%);transition:.2s;z-index:5;display:flex;flex-direction:column}
#side.open{transform:none}#convs{flex:1;overflow-y:auto}.cv{padding:10px 12px;border-bottom:1px solid #20232a;display:flex;gap:6px}.cv span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.cv.cur{background:#1f2430}
#log{flex:1;overflow-y:auto;padding:12px}
.msg{margin:8px 0;padding:10px 12px;border-radius:12px;max-width:100%;white-space:pre-wrap;word-wrap:break-word}
.user{background:var(--acc);color:#fff;margin-left:15%}.bot{background:var(--card)}
.ev{font:12px ui-monospace,monospace;color:var(--mut);margin:3px 0 3px 4px;white-space:pre-wrap;word-break:break-all}
pre{background:#0b0d11;padding:10px;border-radius:8px;overflow-x:auto;font:13px ui-monospace,monospace;white-space:pre}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600;margin-top:6px}
.pass{background:var(--ok)}.fail{background:var(--bad)}
form{display:flex;gap:8px;padding:10px;border-top:1px solid #262a33;padding-bottom:calc(10px + env(safe-area-inset-bottom))}
textarea{flex:1;resize:none;height:46px;border-radius:10px;border:1px solid #333;background:var(--card);color:var(--fg);padding:10px;font:16px system-ui}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:0 16px;font-weight:600}
a{color:#7fb0ff}details summary{cursor:pointer;color:var(--mut);font-size:12px}
</style></head><body>
<div id="side"><div style="padding:10px;display:flex;gap:8px"><button class="hb" id="newc" style="flex:1">+ New chat</button><form method="post" action="/logout" style="padding:0;border:0"><button class="hb">Log out</button></form></div>
<label style="padding:4px 12px;font-size:12px;color:var(--mut)"><input type="checkbox" id="learn"> learn from verified code</label><div id="convs"></div></div>
<header><button class="hb" id="menu">☰</button><b id="ttl">Dolphin</b><button class="hb" id="tts" title="read replies aloud">🔇</button><select id="model"></select></header>
<div id="log"></div>
<form id="f"><button type="button" class="hb" id="mic" title="talk">🎤</button><textarea id="q" placeholder="Message Dolphin…"></textarea><button id="send">Send</button></form>
<script>
const $=id=>document.getElementById(id),log=$('log'),q=$('q');let cur=null,busy=false;
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function md(s){let out='',i=0;const re=/```(\w*)\n([\s\S]*?)```/g;let m;while((m=re.exec(s))){out+=esc(s.slice(i,m.index))+'<pre>'+esc(m[2])+'</pre>';i=re.lastIndex}return out+esc(s.slice(i))}
function add(cls,html){const d=document.createElement('div');d.className=cls;d.innerHTML=html;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
const FRIENDLY={web_search:'🌐 Checking the web…',fetch_url:'🌐 Reading a page…',run_command:'💻 Running a command…',write_file:'📝 Writing a file…',read_file:'📂 Looking at a file…',list_dir:'📂 Looking at files…',run_tests:'🧪 Running tests…',get_trade_signals:'📈 Checking signals…'};
function friendly(e){if(e.type==='tool')return FRIENDLY[e.name]||'🔧 Using '+e.name+'…';if(e.type==='sandbox')return e.passed?'✅ Tested the code, it works':'🧪 Testing the code…';if(e.type==='memory')return '📝 Noted: '+e.fact;if(e.type==='error')return '⚠️ Something went wrong';return ''}
function steps(evs){const raw=evs.map(evText).filter(Boolean),fr=[...new Set(evs.map(friendly).filter(Boolean))];if(!fr.length&&!raw.length)return '';
 return '<details class="ev"><summary>'+esc(fr.join(' · ')||'details')+'</summary>'+raw.join('\n')+'</details>'}
function evText(e){if(e.type==='memory')return '📝 remembered: '+esc(e.fact);if(e.type==='tool')return '🔧 '+e.name+'('+esc(JSON.stringify(e.args).slice(0,300))+')'+(e.auto?' [auto]':'');
 if(e.type==='tool_result')return '   ↳ '+esc(e.preview.replace(/<\/?untrusted[^>]*>/g,'').slice(0,400));
 if(e.type==='sandbox')return '🧪 sandbox run #'+e.attempt+' ('+e.lang+'): '+(e.passed?'PASS':'FAIL');
 if(e.type==='experience')return '📚 recalled past solution '+esc(e.id||'');
 if(e.type==='docs')return '📄 docs: '+esc(e.sources.join(', '));
 if(e.type==='learned')return '🧠 saved to experience library';if(e.type==='error')return '⚠️ '+esc(e.message);return ''}
function botHtml(e){let h=md(e.answer||'');if(e.verdict){h+='<br><span class="badge '+(e.verdict.passed?'pass':'fail')+'">sandbox '+(e.verdict.passed?'PASS':'FAIL')+' ('+e.verdict.lang+', attempt '+e.verdict.attempt+')</span><details><summary>run log</summary><pre>'+esc(e.verdict.log)+'</pre></details>'}
 if(e.sources&&e.sources.length){h+='<div class="ev">'+e.sources.map((s,i)=>'['+(i+1)+'] <a href="'+encodeURI(s.url)+'" target="_blank" rel="noopener noreferrer">'+esc(s.title||s.url)+'</a>').join('<br>')+'</div>'}
 if(e.seconds!=null)h+='<div class="ev">'+e.seconds+'s'+(e.model?' · '+esc(e.model):'')+'</div>';return h}
async function api(p,opt){const r=await fetch(p,opt);if(r.status===401){location.href='/login';throw 0}return r}
async function loadModels(){const d=await (await api('/api/models')).json();$('model').innerHTML=d.models.map(m=>'<option>'+esc(m)+'</option>').join('');$('model').value=localStorage.model||d.models[0]}
$('model').onchange=()=>localStorage.model=$('model').value;
async function loadConvs(){const d=await (await api('/api/convs')).json();$('convs').innerHTML='';
 for(const c of d.convs){const el=document.createElement('div');el.className='cv'+(c.id===cur?' cur':'');el.innerHTML='<span>'+esc(c.title)+'</span><a href="#">✕</a>';
  el.querySelector('span').onclick=()=>{openConv(c.id);$('side').classList.remove('open')};
  el.querySelector('a').onclick=async ev=>{ev.preventDefault();if(!confirm('Delete chat?'))return;await api('/api/conv/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:c.id})});if(c.id===cur)newConv();loadConvs()};
  $('convs').appendChild(el)}}
function newConv(){cur=null;localStorage.removeItem('conv');log.innerHTML='<div class="ev">Hey Colby! Just talk to me. I can also run stuff, write code and search the web when you ask.</div>';$('ttl').textContent='SIC'}
async function openConv(id){const r=await api('/api/conv?id='+encodeURIComponent(id));if(!r.ok){newConv();return}const c=await r.json();cur=c.id;localStorage.conv=cur;$('ttl').textContent=c.title;log.innerHTML='';
 for(const m of c.messages){if(m.role==='user')add('msg user',esc(m.content));else{if(m.events&&m.events.length){const h=steps(m.events);if(h)add('',h)}add('msg bot',botHtml(m))}}loadConvs()}
$('menu').onclick=()=>{$('side').classList.toggle('open');loadConvs()};$('newc').onclick=()=>{newConv();$('side').classList.remove('open')};
$('f').onsubmit=async ev=>{ev.preventDefault();const text=q.value.trim();if(!text||busy)return;busy=true;$('send').disabled=true;q.value='';
 add('msg user',esc(text));const evbox=add('ev','…');let evs=[];const live=add('msg bot','');let lines=[],stream='';
 try{const r=await api('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,conv_id:cur,model:$('model').value,learn:$('learn').checked})});
 const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
 while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let k;
  while((k=buf.indexOf('\n'))>=0){const line=buf.slice(0,k);buf=buf.slice(k+1);if(!line)continue;const e=JSON.parse(line);
   if(e.type==='conv'){cur=e.id;localStorage.conv=cur;$('ttl').textContent=e.title}
   else if(e.type==='token'){stream+=e.text;live.innerHTML=stream.trimStart().startsWith('<tool')?'<span class="ev">calling tool…</span>':md(stream);log.scrollTop=log.scrollHeight}
   else if(e.type==='reset'){stream='';live.innerHTML=''}
   else if(e.type==='final'){live.innerHTML=botHtml(e);if(!evs.length)evbox.remove();speak(e.answer)}
   else{evs.push(e);const h=steps(evs);if(h){evbox.className='';evbox.innerHTML=h}}}}}
 catch(err){if(err)add('ev','⚠️ connection error: '+esc(err))}
 busy=false;$('send').disabled=false;log.scrollTop=log.scrollHeight};
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&window.innerWidth>700){e.preventDefault();$('f').requestSubmit()}});
// voice: Web Speech API (speech-to-text mic, optional text-to-speech)
let ttsOn=localStorage.tts==='1';const ttsB=$('tts');ttsB.textContent=ttsOn?'🔊':'🔇';
ttsB.onclick=()=>{ttsOn=!ttsOn;localStorage.tts=ttsOn?'1':'0';ttsB.textContent=ttsOn?'🔊':'🔇';if(!ttsOn)speechSynthesis.cancel()};
function speak(t){if(!ttsOn||!window.speechSynthesis||!t)return;t=t.replace(/```[\s\S]*?```/g,' (code below) ').replace(/\[\d+\]/g,'').slice(0,1200);speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance(t))}
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;const mic=$('mic');
if(!SR)mic.style.display='none';else{let rec=null;mic.onclick=()=>{if(rec){rec.stop();return}
 rec=new SR();rec.lang=navigator.language||'en-US';rec.interimResults=true;const base=q.value?q.value+' ':'';
 rec.onresult=ev=>{let t='';for(const r of ev.results)t+=r[0].transcript;q.value=base+t;if(ev.results[ev.results.length-1].isFinal&&localStorage.autosend!=='0'){rec.stop();setTimeout(()=>$('f').requestSubmit(),150)}};
 rec.onend=()=>{rec=null;mic.textContent='🎤'};rec.onerror=()=>{rec=null;mic.textContent='🎤'};mic.textContent='⏺';rec.start()}}
loadModels();if(localStorage.conv)openConv(localStorage.conv);else newConv();
</script></body></html>"""

LOGIN = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SIC login</title><style>body{background:#0f1115;color:#eee;font:16px system-ui;display:flex;height:100vh;align-items:center;justify-content:center}
form{display:flex;flex-direction:column;gap:10px;width:260px}input,button{padding:12px;border-radius:10px;border:1px solid #333;font-size:16px}
button{background:#3b82f6;color:#fff;border:0}</style></head><body><form method="post" action="/login">
<b>SIC</b><input type="password" name="password" placeholder="password" autofocus>
<button>Log in</button><small>%MSG%</small></form></body></html>"""


SIGNALS_DIR = os.environ.get("SIC_SIGNALS_DIR", os.path.join(ROOT, "signals", "data"))
SIGNALS_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Signals (paper)</title>
<style>body{margin:0;background:#0f1115;color:#e8e8e8;font:15px/1.4 system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;background:#171a21;padding:10px 12px;border-bottom:1px solid #262a33}
.b{display:inline-block;background:#3a1d1d;color:#ffb4b4;border-radius:6px;padding:2px 6px;font-size:12px}
select,button{font:15px system-ui;background:#222733;color:#eee;border:1px solid #333;border-radius:8px;padding:6px}
.c{background:#171a21;margin:8px;padding:10px;border-radius:10px;border:1px solid #262a33}
.f{border-color:#3d7a3d}.m{color:#9aa3b2;font-size:13px}.t{font-weight:600;word-break:break-all}
table{width:100%;font-size:13px;border-collapse:collapse}td{padding:2px 4px}</style></head><body>
<header><b>Weather signals</b> <span class=b>PAPER ONLY &middot; live trading OFF</span>
<div style="margin-top:6px"><select id=k><option value="">all</option><option>high</option><option>low</option>
<option>rain</option></select> <label><input type=checkbox id=fl checked> flagged</label>
<button onclick=load()>refresh</button> <a href="/" style="color:#8ab4ff">chat</a></div>
<div class=m id=meta></div></header><div id=sc></div><div id=list></div>
<script>
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function load(){const q='?kind='+k.value+'&flagged='+(fl.checked?1:0);
const r=await fetch('/api/signals'+q);if(r.status!==200){location='/login';return}const d=await r.json();
meta.textContent='generated '+(d.generated?new Date(d.generated).toLocaleString():'never')+' \u00b7 '+d.n_signals+' markets \u00b7 '+d.n_flags+' flags';
let s='<div class=c><b>Score vs settled results</b> ('+d.settled_tickers+' settled)<table><tr><td>kind<td>n<td>Brier model<td>Brier mkt<td>paper P&L';
for(const[k2,v]of Object.entries(d.scores||{}))s+='<tr><td>'+esc(k2)+'<td>'+v.tickers+'<td>'+esc(v.brier_model)+'<td>'+esc(v.brier_market)+'<td>'+esc(v.paper_pnl_per_contract_sum)+' ('+v.paper_trades+')';
sc.innerHTML=s+'</table><div class=m>Lower Brier is better. Edge is unproven until model beats market.</div></div>';
list.innerHTML=d.signals.map(x=>'<div class="c'+(x.flag?' f':'')+'"><div class=t>'+esc(x.ticker)+'</div><div class=m>'+esc(x.city)+' \u00b7 '+esc(x.kind)+' \u00b7 '+esc(x.hours_left)+'h left</div>model p(yes) <b>'+esc(x.p_model)+'</b> \u00b7 yes ask '+esc(x.yes_ask)+' \u00b7 no ask '+esc(x.no_ask)+'<br>paper: '+esc(x.side||'-')+' ev '+esc(x.ev)+'</div>').join('')||'<div class=c>no signals</div>'}
k.onchange=load;fl.onchange=load;load();setInterval(load,120000)
</script></body></html>"""


class App:
    def __init__(self, agent_factory, password: str, store: "ConversationStore | None" = None,
                 models: list[str] | None = None, signals_dir: str | None = None):
        self.signals_dir = signals_dir or SIGNALS_DIR
        self.agent_factory = agent_factory  # model name -> agent
        self.agents: dict[str, object] = {}
        self.password = password
        self.store = store or ConversationStore()
        self.models = models or MODELS
        self.sessions: dict[str, float] = {}
        self.failures: list[float] = []
        self.lock = threading.Lock()

    def agent(self, model: str):
        if model not in self.agents:
            self.agents[model] = self.agent_factory(model)
        return self.agents[model]

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

        def _json(self, obj, code=200):
            return self._send(code, json.dumps(obj), "application/json")

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
            if path == "/signals":
                return self._send(200, SIGNALS_PAGE)
            if path == "/api/signals":
                from signals.feed import get_trade_signals
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                return self._json(get_trade_signals(app.signals_dir, kind=(qs.get("kind") or [None])[0],
                                                    flagged_only=(qs.get("flagged") or ["0"])[0] == "1",
                                                    limit=60))
            if path == "/api/models":
                return self._json({"models": app.models})
            if path == "/api/convs":
                return self._json({"convs": app.store.list()})
            if path == "/api/conv":
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                conv = app.store.get((qs.get("id") or [""])[0])
                return self._json(conv) if conv else self._send(404, "not found", "text/plain")
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
            if path not in ("/api/chat", "/api/conv/delete"):
                return self._send(404, "not found", "text/plain")
            if not app.valid(self._cookie()):
                return self._send(401, json.dumps({"error": "auth"}), "application/json")
            try:
                req = json.loads(self._body() or b"{}")
                msg = str(req.get("message", ""))[:8000]
            except json.JSONDecodeError:
                return self._send(400, "bad json", "text/plain")
            if path == "/api/conv/delete":
                app.store.delete(str(req.get("id", "")))
                return self._json({"ok": True})
            model = req.get("model") if req.get("model") in app.models else app.models[0]
            conv = app.store.get(str(req.get("conv_id") or "")) or app.store.new(model)
            if not conv["messages"]:
                conv["title"] = " ".join(msg.split())[:60] or "New chat"
            history = [{"role": m["role"], "content": m["content"]} for m in conv["messages"]]
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True

            events, final = [], {}

            def emit(e):
                if e.get("type") == "final":
                    final.update(e)
                elif e.get("type") not in ("token", "reset"):
                    events.append(e)
                try:
                    self.wfile.write((json.dumps(e) + "\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            emit({"type": "conv", "id": conv["id"], "title": conv["title"]})
            with app.lock:  # single user, one generation at a time
                try:
                    app.agent(model).respond(msg, history, emit=emit, learn=bool(req.get("learn")))
                except Exception as ex:  # report, don't crash the server
                    emit({"type": "error", "message": f"{type(ex).__name__}: {ex}"[:500]})
                    emit({"type": "final", "answer": "Sorry, something went wrong.", "verdict": None,
                          "sources": [], "seconds": 0})
                conv["model"] = model
                conv["messages"].append({"role": "user", "content": msg})
                conv["messages"].append({"role": "assistant", "content": final.get("answer", ""),
                                         "verdict": final.get("verdict"), "sources": final.get("sources", []),
                                         "seconds": final.get("seconds"), "model": model,
                                         "events": [e for e in events if e.get("type") != "conv"][-30:]})
                app.store.save(conv)

    return H


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def default_agent(model: str | None = None):
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
    from .memory import Memory
    llm = OllamaLLM(model=model or MODELS[0], num_predict=1024, temperature=0.7,
                    num_ctx=int(os.environ.get("SIC_NUM_CTX", "8192")), keep_alive="30m")
    llm.options["top_k"] = 40  # natural, less robotic replies (tool turns still parse fine)
    return ChatAgent(llm, sb, AgentComputer(sb), docs=load_store(), experience=lib,
                     signals_dir=SIGNALS_DIR, memory=Memory())


def main():
    host = os.environ.get("SIC_HOST", "127.0.0.1")
    port = int(os.environ.get("SIC_PORT", "8765"))
    app = App(default_agent, load_or_create_password())
    srv = Server((host, port), make_handler(app))
    print(f"SIC chat on http://{host}:{port}  (password file: {PW_FILE})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
