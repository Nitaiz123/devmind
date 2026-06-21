"""
DevMind CLI
============
Command-line interface for DevMind.

Commands:
  devmind trace <script.py>       — trace a Python script and analyze it
  devmind ask <question>          — ask a question about a saved session
  devmind logs <logfile>          — analyze a log file
  devmind serve                   — start the web dashboard
  devmind demo                    — run a built-in demo
"""

import sys
import json
import time
import argparse
from pathlib import Path


def cmd_trace(args):
    """Trace a Python script and analyze it."""
    from devmind.tracer.tracer import Tracer
    from devmind.llm.engine import MockEngine

    script = args.script
    if not Path(script).exists():
        print(f"Error: file '{script}' not found")
        sys.exit(1)

    print(f"🔍 Tracing {script}...")
    tracer = Tracer(label=Path(script).stem, record_lines=args.lines)
    tracer.start()
    try:
        exec(compile(Path(script).read_text(), script, "exec"),
             {"__name__": "__main__", "__file__": script})
    except SystemExit:
        pass
    except Exception as e:
        print(f"⚠️  Script raised: {type(e).__name__}: {e}")
    finally:
        tracer.stop()

    session = tracer.session
    print(f"✅ Captured {len(session.events)} events in {session.duration_ms:.1f}ms")

    if args.save:
        out = args.save
        session.to_json(out)
        print(f"💾 Session saved to {out}")

    if args.analyze:
        print("\n🤖 Analyzing trace with DevMind...\n")
        engine = MockEngine()
        answer = engine.analyze_trace(session)
        print(answer.format())
    else:
        print("\n" + session.timeline_summary())


def cmd_ask(args):
    """Ask a question about a saved session."""
    from devmind.tracer.tracer import TraceSession
    from devmind.infra.analyzer import InfraSession
    from devmind.llm.engine import MockEngine

    engine = MockEngine()
    trace_session = None
    infra_session = None

    if args.trace:
        trace_session = TraceSession.from_json(args.trace)
        print(f"📂 Loaded trace session: {trace_session.label} ({len(trace_session.events)} events)")

    if args.logs:
        infra_session = InfraSession(label=Path(args.logs).stem)
        infra_session.ingest_log_file(args.logs)
        print(f"📂 Loaded log file: {args.logs} ({len(infra_session.logs)} entries)")

    question = " ".join(args.question)
    print(f"\n❓ Question: {question}\n")
    answer = engine.ask(question, trace_session=trace_session, infra_session=infra_session)
    print(answer.format())


def cmd_logs(args):
    """Analyze a log file."""
    from devmind.infra.analyzer import InfraSession
    from devmind.llm.engine import MockEngine

    log_file = args.file
    if not Path(log_file).exists():
        print(f"Error: file '{log_file}' not found")
        sys.exit(1)

    print(f"📥 Loading {log_file}...")
    session = InfraSession(label=Path(log_file).stem)
    session.ingest_log_file(log_file)
    print(f"✅ Parsed {len(session.logs)} log entries")

    errors = [l for l in session.logs if l.is_error()]
    if errors:
        print(f"❌ Found {len(errors)} ERROR/FATAL entries")

    print("\n" + session.timeline_summary())

    if args.analyze:
        print("\n🤖 Analyzing with DevMind...\n")
        engine = MockEngine()
        answer = engine.analyze_infra(session)
        print(answer.format())


def cmd_serve(args):
    """Start the DevMind web server."""
    import uvicorn
    from devmind.api.server import app
    print(f"🚀 Starting DevMind server on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


def cmd_demo(args):
    """Run a built-in demo showing DevMind's capabilities."""
    from devmind.tracer.tracer import Tracer
    from devmind.infra.analyzer import InfraSession
    from devmind.llm.engine import MockEngine

    print("=" * 60)
    print("🧠 DevMind Demo — AI Debugging Copilot")
    print("=" * 60)

    # ── Demo 1: Trace a buggy function ────────────────────────────────────────
    print("\n📌 Demo 1: Tracing a buggy function\n")

    def process_payment(amount: float, balance: float) -> float:
        """Process a payment — has a bug: doesn't check for negative balance."""
        fee = amount * 0.02
        total = amount + fee
        new_balance = balance - total
        return new_balance

    def run_transactions():
        balance = 100.0
        transactions = [30.0, 50.0, 40.0]  # last one will overdraft
        for tx in transactions:
            balance = process_payment(tx, balance)
        return balance

    tracer = Tracer(label="payment_demo")
    with tracer:
        final_balance = run_transactions()

    session = tracer.session
    print(session.timeline_summary(max_events=20))

    engine = MockEngine()
    answer = engine.ask(
        "Why did the balance go negative? What's the bug?",
        trace_session=session,
    )
    print(answer.format())

    # ── Demo 2: Infrastructure log analysis ───────────────────────────────────
    print("\n" + "=" * 60)
    print("📌 Demo 2: Analyzing infrastructure logs\n")

    sample_logs = """
2024-01-15 10:20:00,000 INFO  api.gateway Request received: POST /api/payments
2024-01-15 10:20:00,050 INFO  payment.service Processing payment for user_id=42
2024-01-15 10:20:00,100 INFO  db.pool Connection acquired from pool (pool_size=10/10)
2024-01-15 10:20:05,100 ERROR db.pool Connection timeout after 5000ms
2024-01-15 10:20:05,101 ERROR payment.service Database connection failed: timeout
2024-01-15 10:20:05,102 ERROR api.gateway Request failed: 500 Internal Server Error
2024-01-15 10:20:05,200 ERROR db.pool Connection timeout after 5000ms
2024-01-15 10:20:05,201 ERROR payment.service Database connection failed: timeout
2024-01-15 10:20:05,202 ERROR api.gateway Request failed: 500 Internal Server Error
2024-01-15 10:20:06,000 WARN  db.pool Pool exhausted, waiting for connection
2024-01-15 10:20:06,500 ERROR db.pool Connection timeout after 5000ms
2024-01-15 10:20:10,000 INFO  api.gateway Health check: OK
"""

    infra = InfraSession(label="production-incident")
    infra.ingest_log_text(sample_logs, service="production")
    infra.add_metric("db", "latency_p99", 120, timestamp=time.time() - 600)
    infra.add_metric("db", "latency_p99", 145, timestamp=time.time() - 500)
    infra.add_metric("db", "latency_p99", 890, timestamp=time.time() - 400)
    infra.add_metric("db", "latency_p99", 4800, timestamp=time.time() - 300)
    infra.add_metric("db", "latency_p99", 5100, timestamp=time.time() - 200)
    infra.add_deploy("payment-service", "v2.3.1", commit="a1b2c3d",
                     author="dev@company.com",
                     description="Increase connection pool size",
                     timestamp=time.time() - 700)

    print(infra.timeline_summary())

    answer2 = engine.diagnose_incident(
        infra_session=infra,
        symptom="Payment service returning 500 errors, database timeouts",
    )
    print(answer2.format())

    print("\n" + "=" * 60)
    print("✅ Demo complete! Run `devmind serve` to start the web dashboard.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="devmind",
        description="🧠 DevMind — AI-powered debugging and infrastructure copilot",
    )
    sub = parser.add_subparsers(dest="command")

    # trace
    p_trace = sub.add_parser("trace", help="Trace a Python script")
    p_trace.add_argument("script", help="Python script to trace")
    p_trace.add_argument("--analyze", action="store_true", help="Auto-analyze after tracing")
    p_trace.add_argument("--save", metavar="FILE", help="Save session to JSON file")
    p_trace.add_argument("--lines", action="store_true", help="Record line-level events")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about a session")
    p_ask.add_argument("question", nargs="+", help="Question to ask")
    p_ask.add_argument("--trace", metavar="FILE", help="Trace session JSON file")
    p_ask.add_argument("--logs", metavar="FILE", help="Log file to analyze")

    # logs
    p_logs = sub.add_parser("logs", help="Analyze a log file")
    p_logs.add_argument("file", help="Log file to analyze")
    p_logs.add_argument("--analyze", action="store_true", help="Run AI analysis")

    # serve
    p_serve = sub.add_parser("serve", help="Start the web dashboard")
    p_serve.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")

    # demo
    sub.add_parser("demo", help="Run a built-in demo")

    args = parser.parse_args()

    if args.command == "trace":
        cmd_trace(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "logs":
        cmd_logs(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
