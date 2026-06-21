"""
DevMind REST API Server
========================
Exposes DevMind capabilities over HTTP:
  POST /api/trace/analyze      — analyze a trace session
  POST /api/infra/ingest       — ingest logs/metrics
  POST /api/infra/analyze      — analyze infrastructure data
  POST /api/ask                — ask a natural language question
  POST /api/diagnose           — full incident diagnosis
  GET  /api/sessions           — list active sessions
  GET  /api/health             — health check
  GET  /                       — web dashboard (HTML)
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from devmind.tracer.tracer import TraceSession
from devmind.infra.analyzer import InfraSession
from devmind.llm.engine import MockEngine, DebugAnswer


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DevMind API",
    description="AI-powered debugging and infrastructure analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
_trace_sessions: Dict[str, TraceSession] = {}
_infra_sessions: Dict[str, InfraSession] = {}
_engine = MockEngine()


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    trace_session_id: Optional[str] = None
    infra_session_id: Optional[str] = None
    extra_context: Optional[str] = None


class IngestLogsRequest(BaseModel):
    session_id: Optional[str] = None
    label: Optional[str] = None
    logs: str                       # raw log text
    service: Optional[str] = "unknown"


class AddMetricRequest(BaseModel):
    session_id: str
    service: str
    metric: str
    value: float
    unit: Optional[str] = ""
    timestamp: Optional[float] = None


class AddDeployRequest(BaseModel):
    session_id: str
    service: str
    version: str
    commit: Optional[str] = ""
    author: Optional[str] = ""
    description: Optional[str] = ""
    timestamp: Optional[float] = None


class DiagnoseRequest(BaseModel):
    symptom: str
    trace_session_id: Optional[str] = None
    infra_session_id: Optional[str] = None


class TraceEventRequest(BaseModel):
    session_id: Optional[str] = None
    label: Optional[str] = None
    events: List[Dict[str, Any]]    # list of TraceEvent dicts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_trace(session_id: str) -> TraceSession:
    if session_id not in _trace_sessions:
        raise HTTPException(404, f"Trace session '{session_id}' not found")
    return _trace_sessions[session_id]


def _get_infra(session_id: str) -> InfraSession:
    if session_id not in _infra_sessions:
        raise HTTPException(404, f"Infra session '{session_id}' not found")
    return _infra_sessions[session_id]


def _answer_to_response(answer: DebugAnswer) -> dict:
    return answer.to_dict()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0", "timestamp": time.time()}


@app.get("/api/sessions")
def list_sessions():
    return {
        "trace_sessions": [
            {"id": sid, "label": s.label, "events": len(s.events),
             "duration_ms": s.duration_ms}
            for sid, s in _trace_sessions.items()
        ],
        "infra_sessions": [
            {"id": sid, "label": s.label, "logs": len(s.logs),
             "metrics": len(s.metrics), "deploys": len(s.deploys)}
            for sid, s in _infra_sessions.items()
        ],
    }


@app.post("/api/trace/ingest")
def ingest_trace(req: TraceEventRequest):
    """Ingest a list of trace events into a session."""
    from devmind.tracer.tracer import TraceEvent
    sid = req.session_id or str(uuid.uuid4())[:8]
    if sid not in _trace_sessions:
        _trace_sessions[sid] = TraceSession(sid, label=req.label or sid)
    session = _trace_sessions[sid]
    for ev_data in req.events:
        try:
            ev = TraceEvent(**ev_data)
            session.add_event(ev)
        except Exception:
            pass
    return {"session_id": sid, "events_ingested": len(req.events),
            "total_events": len(session.events)}


@app.post("/api/trace/analyze")
def analyze_trace(session_id: str):
    """Auto-analyze a trace session."""
    session = _get_trace(session_id)
    answer = _engine.analyze_trace(session)
    return _answer_to_response(answer)


@app.post("/api/infra/ingest")
def ingest_logs(req: IngestLogsRequest):
    """Ingest log text into an infra session."""
    sid = req.session_id or str(uuid.uuid4())[:8]
    if sid not in _infra_sessions:
        _infra_sessions[sid] = InfraSession(label=req.label or sid)
    session = _infra_sessions[sid]
    session.ingest_log_text(req.logs, service=req.service or "unknown")
    return {"session_id": sid, "logs_ingested": len(session.logs)}


@app.post("/api/infra/metric")
def add_metric(req: AddMetricRequest):
    """Add a metric data point to an infra session."""
    session = _get_infra(req.session_id)
    session.add_metric(req.service, req.metric, req.value,
                       timestamp=req.timestamp, unit=req.unit or "")
    return {"ok": True, "total_metrics": len(session.metrics)}


@app.post("/api/infra/deploy")
def add_deploy(req: AddDeployRequest):
    """Record a deploy event."""
    session = _get_infra(req.session_id)
    session.add_deploy(req.service, req.version, commit=req.commit or "",
                       author=req.author or "", description=req.description or "",
                       timestamp=req.timestamp)
    return {"ok": True, "total_deploys": len(session.deploys)}


@app.post("/api/infra/analyze")
def analyze_infra(session_id: str):
    """Auto-analyze an infra session."""
    session = _get_infra(session_id)
    answer = _engine.analyze_infra(session)
    return _answer_to_response(answer)


@app.post("/api/ask")
def ask(req: AskRequest):
    """Ask a natural language question about session data."""
    trace_session = _trace_sessions.get(req.trace_session_id) if req.trace_session_id else None
    infra_session = _infra_sessions.get(req.infra_session_id) if req.infra_session_id else None
    answer = _engine.ask(
        req.question,
        trace_session=trace_session,
        infra_session=infra_session,
        extra_context=req.extra_context or "",
    )
    return _answer_to_response(answer)


@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest):
    """Full incident diagnosis combining trace + infra data."""
    trace_session = _trace_sessions.get(req.trace_session_id) if req.trace_session_id else None
    infra_session = _infra_sessions.get(req.infra_session_id) if req.infra_session_id else None
    answer = _engine.diagnose_incident(
        trace_session=trace_session,
        infra_session=infra_session,
        symptom=req.symptom,
    )
    return _answer_to_response(answer)


@app.get("/api/infra/{session_id}/summary")
def infra_summary(session_id: str):
    """Get a structured summary of an infra session."""
    session = _get_infra(session_id)
    return {
        "session_id": session_id,
        "label": session.label,
        "log_count": len(session.logs),
        "error_count": sum(1 for l in session.logs if l.is_error()),
        "error_rate": session.error_rate(),
        "error_clusters": session.error_clusters()[:10],
        "latency_anomalies": session.latency_anomalies()[:10],
        "deploy_correlations": session.deploy_correlations()[:5],
        "timeline": session.timeline_summary(),
    }


@app.get("/api/trace/{session_id}/summary")
def trace_summary(session_id: str):
    """Get a structured summary of a trace session."""
    session = _get_trace(session_id)
    return {
        "session_id": session_id,
        "label": session.label,
        "event_count": len(session.events),
        "duration_ms": session.duration_ms,
        "exceptions": [e.to_dict() for e in session.exceptions()],
        "anomalies": session.anomalies(),
        "hot_functions": session.hot_functions(),
        "call_graph": session.call_graph(),
        "timeline": session.timeline_summary(),
    }


# ── Web Dashboard ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return _DASHBOARD_HTML


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DevMind — AI Debugging Copilot</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --accent: #58a6ff; --green: #3fb950; --red: #f85149;
    --yellow: #d29922; --text: #c9d1d9; --muted: #8b949e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.4rem; color: var(--accent); }
  header span { color: var(--muted); font-size: 0.85rem; }
  .badge { background: var(--accent); color: #000; font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; font-weight: 700; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px 24px; max-width: 1400px; margin: 0 auto; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 0.9rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
  .full-width { grid-column: 1 / -1; }
  textarea, input[type=text] { width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 10px; font-family: monospace; font-size: 0.85rem; resize: vertical; }
  textarea { min-height: 120px; }
  button { background: var(--accent); color: #000; border: none; padding: 8px 18px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; margin-top: 8px; }
  button:hover { opacity: 0.85; }
  button.secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
  .output { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-family: monospace; font-size: 0.8rem; white-space: pre-wrap; min-height: 80px; max-height: 400px; overflow-y: auto; color: var(--text); }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin: 2px; }
  .tag-error { background: rgba(248,81,73,0.15); color: var(--red); }
  .tag-warn { background: rgba(210,153,34,0.15); color: var(--yellow); }
  .tag-ok { background: rgba(63,185,80,0.15); color: var(--green); }
  .stat { text-align: center; padding: 12px; }
  .stat .value { font-size: 2rem; font-weight: 700; color: var(--accent); }
  .stat .label { font-size: 0.75rem; color: var(--muted); margin-top: 4px; }
  .stats-row { display: flex; gap: 8px; }
  .stats-row .stat { flex: 1; background: var(--bg); border-radius: 6px; }
  #chat-messages { min-height: 200px; max-height: 350px; overflow-y: auto; margin-bottom: 8px; }
  .msg { margin: 8px 0; }
  .msg.user { text-align: right; }
  .msg .bubble { display: inline-block; padding: 8px 12px; border-radius: 8px; max-width: 80%; font-size: 0.85rem; }
  .msg.user .bubble { background: var(--accent); color: #000; }
  .msg.ai .bubble { background: var(--border); color: var(--text); text-align: left; }
  .chat-input { display: flex; gap: 8px; }
  .chat-input input { flex: 1; }
  .chat-input button { margin-top: 0; }
  label { font-size: 0.8rem; color: var(--muted); display: block; margin-bottom: 4px; }
  select { background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 8px; font-size: 0.85rem; width: 100%; }
</style>
</head>
<body>
<header>
  <h1>🧠 DevMind</h1>
  <span>AI Debugging Copilot</span>
  <span class="badge">v1.0</span>
</header>

<main>
  <!-- Stats -->
  <div class="card full-width">
    <h2>Session Overview</h2>
    <div class="stats-row">
      <div class="stat"><div class="value" id="stat-trace">0</div><div class="label">Trace Events</div></div>
      <div class="stat"><div class="value" id="stat-logs">0</div><div class="label">Log Entries</div></div>
      <div class="stat"><div class="value" id="stat-errors">0</div><div class="label">Errors</div></div>
      <div class="stat"><div class="value" id="stat-sessions">0</div><div class="label">Sessions</div></div>
    </div>
  </div>

  <!-- Log Ingestion -->
  <div class="card">
    <h2>📥 Ingest Logs</h2>
    <label>Service Name</label>
    <input type="text" id="log-service" placeholder="api-gateway" value="api-gateway" style="margin-bottom:8px">
    <label>Log Text (JSON, Apache, Python, or plain text)</label>
    <textarea id="log-input" placeholder='{"level":"ERROR","message":"Connection timeout","service":"db","latency_ms":5000}
2024-01-15 10:23:45,123 ERROR myapp.db Connection refused to postgres:5432
[10:23:46] ERROR [api] Request failed: 500 Internal Server Error'></textarea>
    <button onclick="ingestLogs()">Ingest Logs</button>
    <div id="log-output" class="output" style="margin-top:8px;display:none"></div>
  </div>

  <!-- Trace Viewer -->
  <div class="card">
    <h2>🔍 Trace Analyzer</h2>
    <label>Paste trace JSON or use the Python SDK to generate one</label>
    <textarea id="trace-input" placeholder='Paste TraceSession JSON here, or use the Python SDK:

from devmind import Tracer
with Tracer(label="my_run") as t:
    my_function()
session_json = t.session.to_json()'></textarea>
    <button onclick="ingestTrace()">Load Trace</button>
    <button class="secondary" onclick="analyzeTrace()" style="margin-left:8px">Auto-Analyze</button>
    <div id="trace-output" class="output" style="margin-top:8px;display:none"></div>
  </div>

  <!-- AI Chat -->
  <div class="card full-width">
    <h2>💬 Ask DevMind</h2>
    <div id="chat-messages"></div>
    <div class="chat-input">
      <input type="text" id="chat-input" placeholder="Why did user.balance go negative? What caused the 500 errors? Which deploy broke production?" onkeydown="if(event.key==='Enter')sendChat()">
      <button onclick="sendChat()">Ask</button>
    </div>
  </div>

  <!-- Incident Diagnosis -->
  <div class="card">
    <h2>🚨 Incident Diagnosis</h2>
    <label>Describe the symptom</label>
    <input type="text" id="symptom-input" placeholder="p99 latency spiked to 5s, error rate 40%" style="margin-bottom:8px">
    <button onclick="diagnose()">Diagnose Incident</button>
    <div id="diagnose-output" class="output" style="margin-top:8px;display:none"></div>
  </div>

  <!-- Session Manager -->
  <div class="card">
    <h2>📂 Active Sessions</h2>
    <button class="secondary" onclick="refreshSessions()">Refresh</button>
    <div id="sessions-output" class="output" style="margin-top:8px"></div>
  </div>
</main>

<script>
let currentTraceId = null;
let currentInfraId = null;

async function api(method, path, body) {
  const opts = { method, headers: {'Content-Type': 'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  return r.json();
}

async function ingestLogs() {
  const logs = document.getElementById('log-input').value;
  const service = document.getElementById('log-service').value || 'unknown';
  if (!logs.trim()) return;
  const res = await api('POST', '/api/infra/ingest', {logs, service, label: service});
  currentInfraId = res.session_id;
  const el = document.getElementById('log-output');
  el.style.display = 'block';
  el.textContent = `✅ Ingested ${res.logs_ingested} log entries → session: ${res.session_id}`;
  refreshStats();
}

async function ingestTrace() {
  const raw = document.getElementById('trace-input').value;
  if (!raw.trim()) return;
  try {
    const data = JSON.parse(raw);
    const events = data.events || [];
    const res = await api('POST', '/api/trace/ingest', {
      events, label: data.label || 'imported', session_id: data.session_id
    });
    currentTraceId = res.session_id;
    const el = document.getElementById('trace-output');
    el.style.display = 'block';
    el.textContent = `✅ Loaded ${res.events_ingested} trace events → session: ${res.session_id}`;
    refreshStats();
  } catch(e) {
    document.getElementById('trace-output').style.display = 'block';
    document.getElementById('trace-output').textContent = `❌ Invalid JSON: ${e.message}`;
  }
}

async function analyzeTrace() {
  if (!currentTraceId) { alert('Load a trace first'); return; }
  const res = await api('POST', `/api/trace/analyze?session_id=${currentTraceId}`);
  const el = document.getElementById('trace-output');
  el.style.display = 'block';
  el.textContent = formatAnswer(res);
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  addMessage('user', q);
  const res = await api('POST', '/api/ask', {
    question: q,
    trace_session_id: currentTraceId,
    infra_session_id: currentInfraId,
  });
  addMessage('ai', formatAnswer(res));
}

async function diagnose() {
  const symptom = document.getElementById('symptom-input').value;
  if (!symptom.trim()) return;
  const res = await api('POST', '/api/diagnose', {
    symptom,
    trace_session_id: currentTraceId,
    infra_session_id: currentInfraId,
  });
  const el = document.getElementById('diagnose-output');
  el.style.display = 'block';
  el.textContent = formatAnswer(res);
}

async function refreshSessions() {
  const res = await api('GET', '/api/sessions');
  const el = document.getElementById('sessions-output');
  let text = 'TRACE SESSIONS:\n';
  for (const s of res.trace_sessions) {
    text += `  ${s.id} | ${s.label} | ${s.events} events | ${s.duration_ms.toFixed(0)}ms\n`;
  }
  text += '\nINFRA SESSIONS:\n';
  for (const s of res.infra_sessions) {
    text += `  ${s.id} | ${s.label} | ${s.logs} logs | ${s.metrics} metrics\n`;
  }
  el.textContent = text || 'No active sessions';
}

async function refreshStats() {
  const res = await api('GET', '/api/sessions');
  let traceEvents = 0, logs = 0, errors = 0;
  for (const s of res.trace_sessions) traceEvents += s.events;
  // Fetch infra summaries
  for (const s of res.infra_sessions) {
    logs += s.logs;
    try {
      const sum = await api('GET', `/api/infra/${s.id}/summary`);
      errors += sum.error_count || 0;
    } catch(e) {}
  }
  document.getElementById('stat-trace').textContent = traceEvents;
  document.getElementById('stat-logs').textContent = logs;
  document.getElementById('stat-errors').textContent = errors;
  document.getElementById('stat-sessions').textContent =
    res.trace_sessions.length + res.infra_sessions.length;
}

function addMessage(role, text) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="bubble">${text.replace(/\\n/g,'<br>')}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function formatAnswer(res) {
  let out = res.answer || '';
  if (res.root_causes && res.root_causes.length > 0) {
    out += '\\n\\nROOT CAUSES:';
    for (const rc of res.root_causes) {
      out += `\\n  [${(rc.confidence*100).toFixed(0)}%] ${rc.title}`;
      out += `\\n  → ${rc.suggested_fix}`;
    }
  }
  return out;
}

// Init
refreshSessions();
refreshStats();
</script>
</body>
</html>"""
