# 🧠 DevMind — AI-Powered Debugging Copilot

> **Stop guessing. Start knowing.** DevMind combines a Python execution Time Machine with an LLM-powered infrastructure analysis engine to find root causes in seconds, not hours.

[![CI](https://github.com/Nitaiz123/devmind/actions/workflows/ci.yml/badge.svg)](https://github.com/Nitaiz123/devmind/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

When something breaks in production, developers spend **hours** doing this:
1. Staring at logs that say `500 Internal Server Error`
2. Adding print statements and redeploying
3. Asking teammates "did you change anything?"
4. Eventually finding it was a sign-flip in `user.balance` caused by a race condition introduced in last Tuesday's deploy

**DevMind eliminates all of that.**

---

## What DevMind Does

### 🕰️ Execution Time Machine
Attach DevMind to any Python process and it records **every function call, variable state, exception, and timing event** into a queryable timeline. Then ask questions in plain English:

```
Why did user.balance go negative at 3am?
Which function is causing the memory leak?
What was the state of the database connection pool when the timeout occurred?
```

### 🔍 Infrastructure Copilot
Feed DevMind your logs, metrics, and deploy history. It automatically:
- Clusters recurring error patterns
- Correlates error spikes with recent deploys
- Detects latency anomalies (p99 spikes, connection pool exhaustion)
- Identifies the blast radius of an incident

### 🤖 LLM Root Cause Analysis
DevMind sends structured context to an LLM (GPT-4, Claude, or local Ollama) and returns:
- **Root causes** with confidence scores and supporting evidence
- **Concrete fix suggestions** with code snippets
- **Follow-up questions** to deepen the investigation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DevMind                              │
├──────────────────┬──────────────────┬───────────────────────┤
│   Execution      │  Infrastructure  │    LLM Engine         │
│   Tracer         │  Analyzer        │                       │
│                  │                  │  ┌─────────────────┐  │
│  sys.settrace()  │  Log Parser      │  │  Context Builder│  │
│  ↓               │  ↓               │  │  ↓              │  │
│  TraceEvent[]    │  LogEntry[]      │  │  Prompt         │  │
│  ↓               │  ↓               │  │  ↓              │  │
│  TraceSession    │  InfraSession    │  │  GPT-4 / Ollama │  │
│  - call graph    │  - error clusters│  │  ↓              │  │
│  - var history   │  - deploy corr.  │  │  RootCause[]    │  │
│  - anomalies     │  - latency spikes│  │  DebugAnswer    │  │
└──────────────────┴──────────────────┴──┴─────────────────┴──┘
         │                  │                    │
         └──────────────────┴────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │         REST API           │
              │  FastAPI + Web Dashboard   │
              └────────────────────────────┘
```

---

## Quick Start

### Install

```bash
pip install devmind
# or from source:
git clone https://github.com/Nitaiz123/devmind
cd devmind && pip install -e .
```

### 1. Trace a Python Script

```python
from devmind import Tracer, MockEngine

def process_payment(amount: float, balance: float) -> float:
    fee = amount * 0.02
    total = amount + fee
    new_balance = balance - total  # Bug: no overdraft check
    return new_balance

# Wrap your code with the tracer
with Tracer(label="payment_run", record_lines=True) as t:
    balance = 100.0
    for tx in [30.0, 50.0, 40.0]:
        balance = process_payment(tx, balance)

# Ask DevMind what went wrong
engine = MockEngine()  # swap for DevMindEngine(model="gpt-4o") with API key
answer = engine.ask(
    "Why did the balance go negative?",
    trace_session=t.session,
)
print(answer.format())
```

**Output:**
```
============================================================
🤖 DevMind Analysis
============================================================

Variable `new_balance` flipped from positive to negative in
process_payment. The function does not check for overdraft
before subtracting the total from the balance.

────────────────────────────────────────────────────────────
📍 Root Causes:

  1. Variable anomaly: new_balance [████████░░] 78%
     Variable `new_balance` flipped from positive (19.6) to
     negative (-21.2) in process_payment:6
     📌 Affected: process_payment
     💡 Fix: Add validation for `new_balance` before it is
             used in process_payment
```

### 2. Analyze Log Files

```bash
# CLI usage
devmind logs /var/log/myapp/app.log --analyze

# Or programmatically
from devmind import InfraSession, MockEngine

session = InfraSession(label="production")
session.ingest_log_file("/var/log/myapp/app.log")
session.add_deploy("api-service", "v2.3.1", commit="a1b2c3d",
                   author="dev@company.com")

engine = MockEngine()
answer = engine.diagnose_incident(
    infra_session=session,
    symptom="Payment service returning 500 errors",
)
print(answer.format())
```

### 3. Web Dashboard

```bash
devmind serve --port 7860
# Open http://localhost:7860
```

The web dashboard lets you:
- Ingest logs by pasting or uploading
- Load trace sessions from JSON
- Ask questions in a chat interface
- Trigger full incident diagnosis

### 4. Run the Built-in Demo

```bash
devmind demo
```

---

## Supported Log Formats

DevMind auto-detects and parses:

| Format | Example |
|--------|---------|
| **JSON** | `{"level":"ERROR","message":"Timeout","latency_ms":5000}` |
| **Python logging** | `2024-01-15 10:23:45,123 ERROR myapp.db Connection refused` |
| **Apache/Nginx** | `192.168.1.1 - - [15/Jan/2024:10:23:45] "GET /api" 500 1234` |
| **Kubernetes** | `2024-01-15T10:23:45.123Z error pod/api-xyz crashed` |
| **Logfmt** | `time=2024-01-15 level=error msg="DB timeout" service=api` |
| **Plain text** | Any line containing ERROR/WARN/FATAL keywords |

---

## Using with a Real LLM

```python
from devmind.llm.engine import DevMindEngine

engine = DevMindEngine(
    model="gpt-4o",
    api_key="sk-...",  # or set OPENAI_API_KEY env var
)

# Multi-turn conversation
answer1 = engine.ask("What caused the 500 errors?", infra_session=session)
print(answer1.format())

answer2 = engine.ask_followup("Which deploy introduced this bug?")
print(answer2.format())

# Works with local Ollama too
engine = DevMindEngine(
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
```

---

## REST API

Start the server with `devmind serve` and use the REST API:

```bash
# Ingest logs
curl -X POST http://localhost:7860/api/infra/ingest \
  -H "Content-Type: application/json" \
  -d '{"logs": "ERROR db Connection timeout\n", "service": "db"}'

# Ask a question
curl -X POST http://localhost:7860/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is causing the timeouts?", "infra_session_id": "abc123"}'

# Full incident diagnosis
curl -X POST http://localhost:7860/api/diagnose \
  -H "Content-Type: application/json" \
  -d '{"symptom": "High error rate after deploy", "infra_session_id": "abc123"}'
```

---

## Tracer API Reference

```python
from devmind import Tracer

# Basic usage
with Tracer(label="my_run") as t:
    my_function()

# With options
tracer = Tracer(
    label="production_trace",
    record_lines=True,          # Record line-level events (more detail, more events)
    max_events=10_000,          # Cap event count to avoid memory issues
    watch_modules=["myapp"],    # Only trace these modules (None = trace everything)
)

# Query the session
session = tracer.session
session.exceptions()            # List all exceptions
session.anomalies()             # Detect sign flips, large jumps
session.hot_functions()         # Most-called functions
session.call_graph()            # Call graph as dict
session.variable_history("x")  # All observed values of variable x
session.timeline_summary()      # Human-readable timeline

# Serialize / deserialize
json_str = session.to_json()
session.to_json("/path/to/file.json")
loaded = TraceSession.from_json("/path/to/file.json")
```

---

## Infrastructure Analyzer API Reference

```python
from devmind import InfraSession

session = InfraSession(label="production")

# Ingest data
session.ingest_log_file("/var/log/app.log")
session.ingest_log_text(log_string, service="api")
session.add_metric("api", "latency_p99", 5000.0, unit="ms")
session.add_deploy("api", "v2.0.0", commit="abc123", author="dev@co.com")

# Analyze
session.error_rate(window_seconds=300)   # Error rate in last 5 minutes
session.error_clusters()                  # Grouped recurring errors
session.latency_anomalies()              # Latency spikes (>3σ)
session.deploy_correlations()            # Errors correlated with deploys
session.timeline_summary()               # Human-readable summary
```

---

## CLI Reference

```
devmind trace <script.py> [--analyze] [--save session.json] [--lines]
devmind ask <question> [--trace session.json] [--logs app.log]
devmind logs <logfile> [--analyze]
devmind serve [--port 7860]
devmind demo
```

---

## Test Suite

```bash
pip install pytest httpx
pytest tests/ -v
# 57 tests covering tracer, log parser, infra analyzer, LLM engine, REST API
```

---

## Roadmap

- [ ] **Distributed tracing** — correlate traces across microservices via trace IDs
- [ ] **VS Code extension** — inline root cause hints in the editor
- [ ] **Prometheus/Grafana integration** — pull metrics directly from monitoring stack
- [ ] **Slack/PagerDuty integration** — auto-diagnose incidents from alerts
- [ ] **Continuous profiling** — always-on sampling profiler with anomaly detection
- [ ] **Go/Java/Node.js agents** — extend tracing beyond Python

---

## License

MIT © 2024 DevMind Contributors
