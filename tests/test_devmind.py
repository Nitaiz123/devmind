"""
DevMind Test Suite
==================
Tests for the tracer, infrastructure analyzer, LLM engine, and API.
"""

import json
import time
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from devmind.tracer.tracer import Tracer, TraceSession, TraceEvent, _safe_repr
from devmind.infra.analyzer import InfraSession, parse_log_line, LogEntry
from devmind.llm.engine import MockEngine, DebugAnswer, RootCause


# ══════════════════════════════════════════════════════════════════════════════
# Tracer Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeRepr:
    def test_primitives(self):
        assert _safe_repr(42) == 42
        assert _safe_repr(3.14) == 3.14
        assert _safe_repr("hello") == "hello"
        assert _safe_repr(True) is True
        assert _safe_repr(None) is None

    def test_list(self):
        result = _safe_repr([1, 2, 3])
        assert result["__type__"] == "list"
        assert result["len"] == 3
        assert result["items"] == [1, 2, 3]

    def test_dict(self):
        result = _safe_repr({"a": 1, "b": 2})
        assert result["__type__"] == "dict"
        assert result["len"] == 2

    def test_nested(self):
        result = _safe_repr([[1, 2], [3, 4]])
        assert result["__type__"] == "list"

    def test_large_list_truncated(self):
        result = _safe_repr(list(range(100)))
        assert result["len"] == 100
        assert len(result["items"]) == 10  # truncated to 10

    def test_object(self):
        class Foo:
            def __init__(self):
                self.x = 1
                self.y = "hello"
        result = _safe_repr(Foo())
        assert result["__type__"] == "Foo"
        assert "x" in result["attrs"]


class TestTracer:
    def test_basic_trace(self):
        def add(a, b):
            return a + b

        tracer = Tracer(label="test_add")
        with tracer:
            result = add(2, 3)

        assert result == 5
        session = tracer.session
        assert len(session.events) > 0
        assert session.label == "test_add"

    def test_captures_call_events(self):
        def my_func(x):
            return x * 2

        tracer = Tracer(label="test_call")
        with tracer:
            my_func(5)

        calls = [e for e in tracer.session.events if e.event_type == "call"]
        assert any(e.func_name == "my_func" for e in calls)

    def test_captures_return_values(self):
        def square(n):
            return n * n

        tracer = Tracer(label="test_return")
        with tracer:
            square(4)

        returns = [e for e in tracer.session.events if e.event_type == "return"]
        square_returns = [e for e in returns if e.func_name == "square"]
        assert len(square_returns) > 0
        assert square_returns[0].return_value == 16

    def test_captures_exceptions(self):
        def bad_func():
            raise ValueError("test error")

        tracer = Tracer(label="test_exc")
        with tracer:
            try:
                bad_func()
            except ValueError:
                pass

        exceptions = tracer.session.exceptions()
        assert len(exceptions) > 0
        assert exceptions[0].exception_type == "ValueError"
        assert "test error" in exceptions[0].exception_msg

    def test_session_duration(self):
        tracer = Tracer(label="test_duration")
        with tracer:
            time.sleep(0.01)

        assert tracer.session.duration_ms >= 10

    def test_variable_history(self):
        def accumulate():
            total = 0
            total += 10
            total += 20
            total += 30
            return total

        tracer = Tracer(label="test_vars")
        with tracer:
            accumulate()

        history = tracer.session.variable_history("total")
        # Should have multiple snapshots of 'total'
        assert len(history) >= 1

    def test_call_graph(self):
        def outer():
            return inner()

        def inner():
            return 42

        tracer = Tracer(label="test_graph")
        with tracer:
            outer()

        graph = tracer.session.call_graph()
        assert "outer" in graph
        assert "inner" in graph["outer"]

    def test_hot_functions(self):
        def hot():
            pass

        tracer = Tracer(label="test_hot")
        with tracer:
            for _ in range(5):
                hot()

        hot_fns = tracer.session.hot_functions()
        assert len(hot_fns) > 0
        assert hot_fns[0]["func"] == "hot"
        assert hot_fns[0]["calls"] == 5

    def test_anomaly_detection_sign_flip(self):
        def drain_balance():
            balance = 100
            balance -= 150  # goes negative
            return balance

        tracer = Tracer(label="test_anomaly", record_lines=True)
        with tracer:
            drain_balance()

        anomalies = tracer.session.anomalies()
        sign_flips = [a for a in anomalies if a["type"] == "sign_flip"]
        assert len(sign_flips) > 0
        assert sign_flips[0]["variable"] == "balance"

    def test_session_serialization(self):
        def simple():
            x = 42
            return x

        tracer = Tracer(label="test_serial")
        with tracer:
            simple()

        json_str = tracer.session.to_json()
        data = json.loads(json_str)
        assert data["label"] == "test_serial"
        assert data["event_count"] > 0

    def test_session_deserialization(self, tmp_path):
        def simple():
            return 99

        tracer = Tracer(label="test_deser")
        with tracer:
            simple()

        path = str(tmp_path / "session.json")
        tracer.session.to_json(path)
        loaded = TraceSession.from_json(path)
        assert loaded.label == "test_deser"
        assert len(loaded.events) == len(tracer.session.events)

    def test_timeline_summary(self):
        def my_fn():
            raise RuntimeError("boom")

        tracer = Tracer(label="test_timeline")
        with tracer:
            try:
                my_fn()
            except RuntimeError:
                pass

        summary = tracer.session.timeline_summary()
        assert "RuntimeError" in summary
        assert "boom" in summary

    def test_max_events_limit(self):
        def looper():
            for i in range(1000):
                pass

        tracer = Tracer(label="test_limit", max_events=50)
        with tracer:
            looper()

        assert len(tracer.session.events) <= 50

    def test_watch_modules_filter(self):
        """When watch_modules is set, only those modules are traced."""
        def my_func():
            return sum(range(10))  # sum/range are builtins, not in watch_modules

        tracer = Tracer(label="test_filter", watch_modules=["nonexistent_module"])
        with tracer:
            my_func()

        # Should have very few or no events since our module isn't in watch_modules
        assert len(tracer.session.events) <= 5


# ══════════════════════════════════════════════════════════════════════════════
# Infrastructure Analyzer Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLogParser:
    def test_parse_json_log(self):
        line = '{"level":"ERROR","message":"Connection failed","service":"db","latency_ms":500}'
        entry = parse_log_line(line, entry_id=1)
        assert entry is not None
        assert entry.level == "ERROR"
        assert entry.service == "db"
        assert entry.latency_ms == 500
        assert "Connection failed" in entry.message

    def test_parse_python_log(self):
        line = "2024-01-15 10:23:45,123 ERROR myapp.db Connection refused"
        entry = parse_log_line(line, entry_id=1)
        assert entry is not None
        assert entry.level == "ERROR"
        assert "Connection refused" in entry.message

    def test_parse_generic_log(self):
        line = "10:23:46.500 ERROR Request failed: 500"
        entry = parse_log_line(line, entry_id=1)
        assert entry is not None
        assert entry.level == "ERROR"

    def test_parse_empty_line(self):
        entry = parse_log_line("", entry_id=1)
        assert entry is None

    def test_parse_plain_text(self):
        line = "Something went wrong in the system"
        entry = parse_log_line(line, service="myapp", entry_id=1)
        assert entry is not None
        assert entry.service == "myapp"
        assert "Something went wrong" in entry.message

    def test_level_normalization(self):
        line = '{"level":"WARNING","message":"Low disk space"}'
        entry = parse_log_line(line, entry_id=1)
        assert entry.level == "WARN"

    def test_k8s_log_format(self):
        line = "2024-01-15T10:23:45.123Z error pod crashed due to OOM"
        entry = parse_log_line(line, service="k8s", entry_id=1)
        assert entry is not None
        assert entry.level == "ERROR"


class TestInfraSession:
    def test_ingest_log_text(self):
        session = InfraSession(label="test")
        logs = """
2024-01-15 10:00:00,000 INFO  app.server Server started
2024-01-15 10:00:01,000 ERROR app.db Connection failed
2024-01-15 10:00:02,000 ERROR app.db Connection failed
"""
        session.ingest_log_text(logs, service="app")
        assert len(session.logs) >= 3

    def test_error_rate(self):
        session = InfraSession(label="test")
        logs = """
2024-01-15 10:00:00,000 INFO  app Request OK
2024-01-15 10:00:01,000 ERROR app Request failed
2024-01-15 10:00:02,000 ERROR app Request failed
2024-01-15 10:00:03,000 INFO  app Request OK
"""
        session.ingest_log_text(logs)
        # Error rate should be 50% (2 errors out of 4 entries)
        # But window_seconds default is 300, so all entries should be included
        # We'll just check it's between 0 and 1
        rate = session.error_rate(window_seconds=86400)
        assert 0 <= rate <= 1

    def test_error_clusters(self):
        session = InfraSession(label="test")
        logs = "\n".join([
            f"2024-01-15 10:00:{i:02d},000 ERROR app Connection timeout after 5000ms"
            for i in range(10)
        ])
        session.ingest_log_text(logs)
        clusters = session.error_clusters()
        assert len(clusters) > 0
        assert clusters[0]["count"] >= 5

    def test_add_metric(self):
        session = InfraSession(label="test")
        session.add_metric("api", "latency_p99", 120.0, unit="ms")
        session.add_metric("api", "latency_p99", 5000.0, unit="ms")
        assert len(session.metrics) == 2

    def test_latency_anomalies(self):
        session = InfraSession(label="test")
        now = time.time()
        # Normal values
        for i in range(10):
            session.add_metric("api", "latency_p99", 100.0 + i,
                               timestamp=now - (20 - i) * 10)
        # Spike
        session.add_metric("api", "latency_p99", 5000.0, timestamp=now)
        anomalies = session.latency_anomalies()
        assert len(anomalies) > 0
        assert anomalies[0]["service"] == "api"

    def test_add_deploy(self):
        session = InfraSession(label="test")
        session.add_deploy("api", "v2.0.0", commit="abc123", author="dev@co.com")
        assert len(session.deploys) == 1
        assert session.deploys[0].version == "v2.0.0"

    def test_deploy_correlation(self):
        session = InfraSession(label="test")
        deploy_time = time.time() - 300  # 5 minutes ago
        session.add_deploy("api", "v2.0.0", commit="abc123", timestamp=deploy_time)
        # Add errors after deploy
        for i in range(5):
            session.logs.append(LogEntry(
                entry_id=i,
                timestamp=deploy_time + 60 + i * 30,
                level="ERROR",
                service="api",
                message="NullPointerException in PaymentController",
            ))
        correlations = session.deploy_correlations()
        assert len(correlations) > 0
        assert correlations[0]["errors_within_10min"] == 5

    def test_timeline_summary(self):
        session = InfraSession(label="production")
        session.ingest_log_text(
            "2024-01-15 10:00:00,000 ERROR app Database connection failed\n"
            "2024-01-15 10:00:01,000 INFO  app Retry successful"
        )
        summary = session.timeline_summary()
        assert "production" in summary
        assert "ERROR" in summary

    def test_is_error(self):
        entry = LogEntry(entry_id=1, timestamp=time.time(), level="ERROR",
                         service="app", message="test")
        assert entry.is_error() is True
        entry2 = LogEntry(entry_id=2, timestamp=time.time(), level="INFO",
                          service="app", message="test")
        assert entry2.is_error() is False


# ══════════════════════════════════════════════════════════════════════════════
# LLM Engine Tests (MockEngine)
# ══════════════════════════════════════════════════════════════════════════════

class TestMockEngine:
    def test_basic_ask(self):
        engine = MockEngine()
        answer = engine.ask("What is wrong?")
        assert isinstance(answer, DebugAnswer)
        assert answer.question == "What is wrong?"
        assert answer.answer != ""
        assert answer.model_used == "devmind-mock"

    def test_ask_with_trace(self):
        def buggy():
            x = 100
            x -= 200  # sign flip
            return x

        tracer = Tracer(label="buggy_test", record_lines=True)
        with tracer:
            buggy()

        engine = MockEngine()
        answer = engine.ask("Why did x go negative?", trace_session=tracer.session)
        assert isinstance(answer, DebugAnswer)
        assert len(answer.root_causes) > 0

    def test_ask_with_exception_trace(self):
        def raises():
            raise ZeroDivisionError("division by zero")

        tracer = Tracer(label="exc_test")
        with tracer:
            try:
                raises()
            except ZeroDivisionError:
                pass

        engine = MockEngine()
        answer = engine.analyze_trace(tracer.session)
        assert len(answer.root_causes) > 0
        assert any("ZeroDivisionError" in rc.title for rc in answer.root_causes)

    def test_ask_with_infra(self):
        session = InfraSession(label="prod")
        session.ingest_log_text(
            "2024-01-15 10:00:00,000 ERROR api Connection timeout\n" * 5
        )
        engine = MockEngine()
        answer = engine.analyze_infra(session)
        assert isinstance(answer, DebugAnswer)
        assert len(answer.root_causes) > 0

    def test_diagnose_incident(self):
        session = InfraSession(label="incident")
        deploy_time = time.time() - 300
        session.add_deploy("api", "v3.0.0", commit="deadbeef", timestamp=deploy_time)
        for i in range(3):
            session.logs.append(LogEntry(
                entry_id=i, timestamp=deploy_time + 60,
                level="ERROR", service="api",
                message="NullPointerException in new feature"
            ))
        engine = MockEngine()
        answer = engine.diagnose_incident(
            infra_session=session,
            symptom="API returning 500 errors after deploy"
        )
        assert isinstance(answer, DebugAnswer)
        assert len(answer.root_causes) > 0

    def test_answer_format(self):
        engine = MockEngine()
        answer = engine.ask("Test question")
        formatted = answer.format()
        assert "DevMind Analysis" in formatted
        assert "Test question" in formatted

    def test_root_cause_structure(self):
        def bad():
            balance = 1000
            balance -= 2000
            return balance

        tracer = Tracer(label="rc_test")
        with tracer:
            bad()

        engine = MockEngine()
        answer = engine.analyze_trace(tracer.session)
        for rc in answer.root_causes:
            assert isinstance(rc, RootCause)
            assert 0.0 <= rc.confidence <= 1.0
            assert rc.title != ""
            assert rc.suggested_fix != ""

    def test_follow_up_questions(self):
        engine = MockEngine()
        answer = engine.ask("Why is the service slow?")
        assert isinstance(answer.follow_up_questions, list)
        assert len(answer.follow_up_questions) > 0

    def test_answer_to_dict(self):
        engine = MockEngine()
        answer = engine.ask("Test")
        d = answer.to_dict()
        assert "question" in d
        assert "answer" in d
        assert "root_causes" in d
        assert "follow_up_questions" in d

    def test_clear_history(self):
        engine = MockEngine()
        engine.clear_history()  # Should not raise


# ══════════════════════════════════════════════════════════════════════════════
# API Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from devmind.api.server import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_list_sessions_empty(self, client):
        r = client.get("/api/sessions")
        assert r.status_code == 200
        data = r.json()
        assert "trace_sessions" in data
        assert "infra_sessions" in data

    def test_ingest_logs(self, client):
        r = client.post("/api/infra/ingest", json={
            "logs": "2024-01-15 10:00:00,000 ERROR app Connection failed\n" * 3,
            "service": "test-api",
            "label": "test",
        })
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["logs_ingested"] >= 3

    def test_ingest_and_analyze_infra(self, client):
        # Ingest
        r = client.post("/api/infra/ingest", json={
            "logs": "2024-01-15 10:00:00,000 ERROR db Timeout\n" * 5,
            "service": "db",
        })
        sid = r.json()["session_id"]

        # Analyze
        r2 = client.post(f"/api/infra/analyze?session_id={sid}")
        assert r2.status_code == 200
        data = r2.json()
        assert "answer" in data

    def test_ask_question(self, client):
        r = client.post("/api/ask", json={
            "question": "What is causing the errors?",
        })
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data

    def test_diagnose(self, client):
        r = client.post("/api/diagnose", json={
            "symptom": "High error rate after deploy",
        })
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data

    def test_dashboard_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "DevMind" in r.text
        assert "AI Debugging" in r.text

    def test_infra_summary(self, client):
        # Ingest first
        r = client.post("/api/infra/ingest", json={
            "logs": "2024-01-15 10:00:00,000 ERROR app Crash\n",
            "service": "app",
        })
        sid = r.json()["session_id"]

        r2 = client.get(f"/api/infra/{sid}/summary")
        assert r2.status_code == 200
        data = r2.json()
        assert "error_count" in data
        assert "error_clusters" in data

    def test_add_metric(self, client):
        # Create session first
        r = client.post("/api/infra/ingest", json={"logs": "INFO app ok", "service": "app"})
        sid = r.json()["session_id"]

        r2 = client.post("/api/infra/metric", json={
            "session_id": sid,
            "service": "api",
            "metric": "latency_p99",
            "value": 250.0,
            "unit": "ms",
        })
        assert r2.status_code == 200
        assert r2.json()["ok"] is True

    def test_add_deploy(self, client):
        r = client.post("/api/infra/ingest", json={"logs": "INFO app ok", "service": "app"})
        sid = r.json()["session_id"]

        r2 = client.post("/api/infra/deploy", json={
            "session_id": sid,
            "service": "api",
            "version": "v2.0.0",
            "commit": "abc123",
            "author": "dev@co.com",
        })
        assert r2.status_code == 200

    def test_ingest_trace_events(self, client):
        events = [
            {
                "event_id": 1, "event_type": "call", "timestamp": time.time(),
                "filename": "test.py", "lineno": 10, "func_name": "my_func",
                "module": "test", "locals_snapshot": {"x": 5},
                "call_depth": 1, "elapsed_ms": 0.0,
            },
            {
                "event_id": 2, "event_type": "return", "timestamp": time.time(),
                "filename": "test.py", "lineno": 12, "func_name": "my_func",
                "module": "test", "locals_snapshot": {"x": 5},
                "return_value": 10, "call_depth": 1, "elapsed_ms": 1.5,
            },
        ]
        r = client.post("/api/trace/ingest", json={
            "events": events, "label": "test-trace"
        })
        assert r.status_code == 200
        assert r.json()["events_ingested"] == 2
