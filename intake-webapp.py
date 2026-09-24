"""
Knowledge Universe — hosted intake web app (first-round testing).

A single FastAPI process that serves both the API and a minimal browser UI.
Colleagues visit a URL, enter a shared password, have the intake conversation,
review the provenance-tagged confirmation, and submit a schema-valid entry that
lands in ./entries/ for you to review.

Design for THIS round (2-5 trusted testers, your key covers everyone):
  - Your LLM key lives ONLY server-side (env var). Never sent to the browser.
  - One shared password (env var) gates access. Do not publicly link the URL.
  - Per-session rate limit + max message length guard your API bill.
  - Entries written as files; no database, no user accounts.

Run:
  pip install fastapi uvicorn pydantic anthropic   # or openai
  export KU_LLM=anthropic
  export ANTHROPIC_API_KEY=sk-...
  export KU_APP_PASSWORD=choose-a-shared-secret
  uvicorn intake_app:app --host 0.0.0.0 --port 8000

Deploy: any host that runs a Python process (Render, Railway, Fly.io, a small
VM). Set the same env vars in the host's dashboard. Share the URL + password
privately with your testers.
"""
import os, json, time, uuid, re
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ----------------------------------------------------------------------------
# Config & guardrails
# ----------------------------------------------------------------------------
APP_PASSWORD = os.getenv("KU_APP_PASSWORD")           # required
LLM_BACKEND  = os.getenv("KU_LLM", "anthropic")
MAX_MSG_CHARS = 8000                                   # bill guard: no novels
MAX_TURNS_PER_SESSION = 40                             # bill guard: no runaways
ENTRIES_DIR = os.getenv("KU_ENTRIES_DIR", "entries")

if not APP_PASSWORD:
    raise RuntimeError("Set KU_APP_PASSWORD before starting.")

# ----------------------------------------------------------------------------
# LLM seam (server-side only; key never leaves the server)
# ----------------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    if LLM_BACKEND == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(
            model=os.getenv("KU_LLM_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=1500, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in r.content if hasattr(b, "text"))
    if LLM_BACKEND == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))
        r = client.chat.completions.create(
            model=os.getenv("KU_LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        return r.choices[0].message.content
    raise RuntimeError(f"Unknown KU_LLM backend: {LLM_BACKEND}")

# ----------------------------------------------------------------------------
# Extraction rules (same discipline as the CLI intake agent)
# ----------------------------------------------------------------------------
EXTRACTION_RULES = """You are an intake agent for a database of NEGATIVE and
inconclusive scientific results. The database's ONLY value is its HONESTY.

Convert what the scientist tells you into structured fields by EXTRACTION ONLY.
You are rewarded for recording gaps and penalized for filling them.

1. NEVER invent a value they didn't state. If unsure, mark it 'unknown'. An honest
   'unknown' is ALWAYS better than a guess.
2. Do not normalize away uncertainty (keep "standard buffer" as-is; don't invent
   concentrations).
3. Classify every field: 'stated' (quote their words), 'inferred' (give reasoning,
   shown for confirmation), or 'unknown'.
4. Ask about HIGH-VALUE gaps specifically, one or two at a time: evidence layer
   (binding/functional/phenotypic/etc.), positive control (run? worked?),
   replicates/sample size, and interpretation-critical conditions.
5. NEVER propose the confidence level or write caveats for them — prompt them to
   state these in their own words. That is their judgment, not yours.
6. Be brief. This is a working scientist pasting notes."""

FIELDS = ["claim_type","evidence_layer","domain","title","observation","conditions",
          "conditions_structured","caveats","alternatives","system","method",
          "confidence_level","positive_control","powered","n","detection_limit",
          "controls","related_positive","contributor","date","references"]
HUMAN_OWNED = {"confidence_level", "caveats"}
HIGH_VALUE  = ["evidence_layer","positive_control","confidence_level","powered","n",
               "conditions_structured"]

# ----------------------------------------------------------------------------
# In-memory session store (fine for a handful of testers; resets on restart)
# ----------------------------------------------------------------------------
class Session:
    def __init__(self):
        self.fields: Dict[str, dict] = {k: {"value": None, "provenance": "unknown",
                                            "source": None} for k in FIELDS}
        self.turns = 0
        self.history: List[dict] = []

SESSIONS: Dict[str, Session] = {}

def check_auth(x_ku_password: Optional[str] = Header(None)):
    if x_ku_password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Bad or missing password.")

# ----------------------------------------------------------------------------
# Core intake logic
# ----------------------------------------------------------------------------
def parse_turn(sess: Session, text: str):
    prompt = (f"{EXTRACTION_RULES}\n\nCurrent fields:\n"
              f"{json.dumps(sess.fields, indent=2)}\n\n"
              f'The scientist said:\n"""{text}"""\n\n'
              "Return ONLY a JSON object of fields to update, each as "
              '{"value":..., "provenance":"stated|inferred|unknown", "source":"..."}. '
              "Include a field ONLY if this turn changes it. Never invent values.")
    raw = call_llm(prompt)
    try:
        updates = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except Exception:
        return  # parse failure changes nothing — never guess to recover
    for name, u in updates.items():
        if name not in sess.fields:
            continue
        prov = u.get("provenance", "unknown")
        if name in HUMAN_OWNED and prov == "inferred":
            continue  # human-owned fields can't be auto-filled by inference
        sess.fields[name] = {"value": u.get("value"), "provenance": prov,
                             "source": u.get("source")}

def next_question(sess: Session) -> Optional[str]:
    gaps = [k for k in HIGH_VALUE
            if sess.fields[k]["provenance"] in ("unknown", "inferred")]
    if not gaps:
        return None
    prompt = (f"{EXTRACTION_RULES}\n\nCurrent fields:\n"
              f"{json.dumps(sess.fields, indent=2)}\n\n"
              f"Unresolved high-value gaps: {gaps}. Ask ONE brief, plain question "
              "about the most important. If it's confidence_level or caveats, PROMPT "
              "them to state it themselves — do not propose a value.")
    return call_llm(prompt)

def confirmation(sess: Session) -> str:
    prompt = (f"{EXTRACTION_RULES}\n\nProduce a confirmation summary in THREE clearly "
              "separated sections:\nSTATED (from their words):\nINFERRED (my reading — "
              "confirm/correct each):\nSTILL UNKNOWN (honestly blank):\nNever hide "
              "inferred items inside stated. Then ask them to confirm, correct, or fill.\n\n"
              f"Fields:\n{json.dumps(sess.fields, indent=2)}")
    return call_llm(prompt)

def to_entry_yaml(sess: Session) -> str:
    import yaml
    def v(name, default=None):
        f = sess.fields[name]
        return default if f["provenance"] == "unknown" or f["value"] is None else f["value"]
    entry = {
        "id": "ku-pending-00000000", "version": "0.1.0",
        "claim_type": v("claim_type", "real_null"),
        "evidence_layer": v("evidence_layer"),
        "domain": v("domain", "unspecified"),
        "title": v("title", "Untitled negative result"),
        "observation": v("observation", ""), "conditions": v("conditions", ""),
        "caveats": v("caveats", ""), "alternatives": v("alternatives", ""),
        "system": v("system", {}), "method": v("method", {"name": "unspecified"}),
        "confidence": {
            "level": v("confidence_level", "low"),
            "positive_control": v("positive_control",
                {"present": False, "detail": "not recorded during intake"}),
            "powered": v("powered"), "controls": v("controls", ""),
            "n": v("n", ""), "detection_limit": v("detection_limit", ""),
        },
        "related_positive": v("related_positive", []),
        "contributor": v("contributor", {"name": "unknown"}),
        "date": v("date", ""), "references": v("references", []),
    }
    unknowns = [k for k in FIELDS if sess.fields[k]["provenance"] == "unknown"]
    header = ("# Generated by hosted conversational intake.\n"
              f"# Fields left UNKNOWN (honestly blank): {unknowns}\n\n")
    return header + yaml.safe_dump(entry, sort_keys=False)

# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
app = FastAPI(title="Knowledge Universe — Intake")

class StartResp(BaseModel):
    session_id: str

class MsgReq(BaseModel):
    session_id: str
    message: str

class MsgResp(BaseModel):
    agent: str
    ready_for_confirmation: bool

@app.post("/api/start", response_model=StartResp)
def start(_: None = Depends(check_auth)):
    sid = uuid.uuid4().hex
    SESSIONS[sid] = Session()
    return StartResp(session_id=sid)

@app.post("/api/message", response_model=MsgResp)
def message(req: MsgReq, _: None = Depends(check_auth)):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Unknown session — start a new one.")
    if sess.turns >= MAX_TURNS_PER_SESSION:
        raise HTTPException(429, "Session turn limit reached — please submit or restart.")
    if len(req.message) > MAX_MSG_CHARS:
        raise HTTPException(413, f"Message too long (max {MAX_MSG_CHARS} chars).")
    sess.turns += 1
    parse_turn(sess, req.message.strip())
    q = next_question(sess)
    if q is None:
        return MsgResp(agent="I have the high-value fields. Click 'Review & confirm' "
                             "when ready, or add more detail.", ready_for_confirmation=True)
    return MsgResp(agent=q, ready_for_confirmation=False)

@app.post("/api/confirm")
def confirm(req: MsgReq, _: None = Depends(check_auth)):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Unknown session.")
    return {"summary": confirmation(sess)}

@app.post("/api/submit")
def submit(req: MsgReq, _: None = Depends(check_auth)):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Unknown session.")
    os.makedirs(ENTRIES_DIR, exist_ok=True)
    safe = re.sub(r"[^a-z0-9]+", "-", (req.message or "draft").lower())[:30] or "draft"
    path = os.path.join(ENTRIES_DIR, f"intake-{safe}-{sess.turns}-{int(time.time())}.yaml")
    yaml_text = to_entry_yaml(sess)
    with open(path, "w") as f:
        f.write(yaml_text)
    return {"written": path, "yaml": yaml_text}

# ----------------------------------------------------------------------------
# Minimal browser UI (single page; password kept in-memory client-side only)
# ----------------------------------------------------------------------------
PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Knowledge Universe — Intake</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
 h1{font-size:1.3rem} .muted{color:#666;font-size:.9rem}
 #log{border:1px solid #ddd;border-radius:8px;padding:1rem;min-height:240px;margin:1rem 0;white-space:pre-wrap}
 .agent{color:#0a5} .you{color:#333;font-weight:600} .sys{color:#a50}
 textarea{width:100%;height:80px;font:inherit;padding:.5rem;box-sizing:border-box}
 button{font:inherit;padding:.5rem 1rem;margin:.25rem .25rem 0 0;cursor:pointer}
 input{font:inherit;padding:.4rem;width:100%;box-sizing:border-box}
 pre{background:#f6f6f6;padding:1rem;border-radius:8px;overflow:auto}
</style></head><body>
<h1>Knowledge Universe — Intake</h1>
<p class=muted>Paste your lab notes or describe a negative/inconclusive result. The
agent extracts structured fields <b>without inventing anything</b>, asks about the
gaps that matter, then shows you exactly what it recorded before you submit.</p>

<div id=gate>
  <p>Enter the shared password to begin:</p>
  <input id=pw type=password placeholder="password">
  <button onclick=begin()>Start</button>
</div>

<div id=app style=display:none>
  <div id=log></div>
  <textarea id=msg placeholder="Describe your result, or paste notes..."></textarea><br>
  <button onclick=send()>Send</button>
  <button onclick=review()>Review &amp; confirm</button>
  <button onclick=submit()>Submit entry</button>
  <div id=out></div>
</div>

<script>
let PW="", SID="";
const log=document.getElementById('log'), out=document.getElementById('out');
function add(who,txt,cls){const d=document.createElement('div');d.className=cls;
  d.textContent=(who?who+": ":"")+txt;log.appendChild(d);log.scrollTop=log.scrollHeight;}
async function api(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json',
    'X-KU-Password':PW},body:JSON.stringify(body||{})});
  if(!r.ok){add("system","Error: "+(await r.text()),"sys");throw new Error(r.status);}
  return r.json();
}
async function begin(){
  PW=document.getElementById('pw').value;
  try{const s=await api('/api/start');SID=s.session_id;
    document.getElementById('gate').style.display='none';
    document.getElementById('app').style.display='block';
    add("agent","Tell me about your result — paste notes or just describe it.","agent");
  }catch(e){}
}
async function send(){
  const m=document.getElementById('msg').value.trim();if(!m)return;
  add("you",m,"you");document.getElementById('msg').value='';
  const r=await api('/api/message',{session_id:SID,message:m});
  add("agent",r.agent,"agent");
}
async function review(){
  const r=await api('/api/confirm',{session_id:SID,message:''});
  out.innerHTML='<h3>Confirmation — check provenance before submitting</h3><pre>'
    +r.summary.replace(/</g,'&lt;')+'</pre>';
}
async function submit(){
  const r=await api('/api/submit',{session_id:SID,message:''});
  out.innerHTML='<h3>Submitted ✓</h3><p class=muted>Saved for review. Copy below if you '
    +'want to keep it:</p><pre>'+r.yaml.replace(/</g,'&lt;')+'</pre>';
}
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
