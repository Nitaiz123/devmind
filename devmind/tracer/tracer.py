"""
DevMind Execution Tracer
========================
A Python sys.settrace-based execution recorder that captures:
- Every function call and return with arguments and return values
- Variable state snapshots at each line
- Exception events with full stack context
- Timing information per frame

The resulting TraceSession is a queryable, serializable timeline
that the LLM engine can reason over.
"""

import sys
import time
import threading
import traceback
import inspect
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone
from pathlib import Path
import copy


# ── Value snapshot ────────────────────────────────────────────────────────────

def _safe_repr(value: Any, depth: int = 0) -> Any:
    """Produce a JSON-serialisable snapshot of a value (max depth 3)."""
    if depth > 3:
        return "<...>"
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        t = "list" if isinstance(value, list) else "tuple"
        return {"__type__": t, "len": len(value),
                "items": [_safe_repr(v, depth + 1) for v in value[:10]]}
    if isinstance(value, dict):
        return {"__type__": "dict", "len": len(value),
                "items": {str(k): _safe_repr(v, depth + 1)
                          for k, v in list(value.items())[:10]}}
    if isinstance(value, set):
        return {"__type__": "set", "len": len(value),
                "items": [_safe_repr(v, depth + 1) for v in list(value)[:10]]}
    try:
        cls = type(value).__name__
        attrs = {}
        for attr in list(vars(value))[:10]:
            try:
                attrs[attr] = _safe_repr(getattr(value, attr), depth + 1)
            except Exception:
                attrs[attr] = "<error>"
        return {"__type__": cls, "attrs": attrs}
    except Exception:
        try:
            return repr(value)[:200]
        except Exception:
            return "<unrepresentable>"


# ── Event model ───────────────────────────────────────────────────────────────

@dataclass
class TraceEvent:
    event_id: int
    event_type: str          # "call" | "return" | "line" | "exception"
    timestamp: float
    filename: str
    lineno: int
    func_name: str
    module: str
    locals_snapshot: Dict[str, Any]
    return_value: Any = None
    exception_type: Optional[str] = None
    exception_msg: Optional[str] = None
    exception_tb: Optional[str] = None
    call_depth: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        ts = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        if self.event_type == "call":
            args = ", ".join(f"{k}={repr(v)[:40]}" for k, v in self.locals_snapshot.items())
            return f"[{ts}] {'  ' * self.call_depth}CALL {self.func_name}({args}) @ {Path(self.filename).name}:{self.lineno}"
        elif self.event_type == "return":
            return f"[{ts}] {'  ' * self.call_depth}RETURN {self.func_name} → {repr(self.return_value)[:60]}"
        elif self.event_type == "exception":
            return f"[{ts}] {'  ' * self.call_depth}EXCEPTION {self.exception_type}: {self.exception_msg} @ {Path(self.filename).name}:{self.lineno}"
        else:
            changed = list(self.locals_snapshot.keys())[:3]
            return f"[{ts}] {'  ' * self.call_depth}LINE {Path(self.filename).name}:{self.lineno} vars={changed}"


# ── Trace session ─────────────────────────────────────────────────────────────

class TraceSession:
    """A complete recorded execution session."""

    def __init__(self, session_id: str, label: str = ""):
        self.session_id = session_id
        self.label = label
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.events: List[TraceEvent] = []
        self._lock = threading.Lock()

    def add_event(self, event: TraceEvent):
        with self._lock:
            self.events.append(event)

    def finish(self):
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    # ── Query API ──────────────────────────────────────────────────────────────

    def exceptions(self) -> List[TraceEvent]:
        return [e for e in self.events if e.event_type == "exception"]

    def calls_to(self, func_name: str) -> List[TraceEvent]:
        return [e for e in self.events
                if e.event_type == "call" and e.func_name == func_name]

    def variable_history(self, var_name: str) -> List[Dict]:
        """Return all recorded values of a variable over time."""
        history = []
        for event in self.events:
            if var_name in event.locals_snapshot:
                history.append({
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "func": event.func_name,
                    "lineno": event.lineno,
                    "value": event.locals_snapshot[var_name],
                })
        return history

    def anomalies(self) -> List[Dict]:
        """Detect suspicious patterns: value sign flips, None→value→None, etc."""
        issues = []
        # Track numeric variable sign changes across all events
        num_history: Dict[str, list] = {}
        for event in self.events:
            for k, v in event.locals_snapshot.items():
                if isinstance(v, (int, float)):
                    if k not in num_history:
                        num_history[k] = []
                    num_history[k].append((event.event_id, v, event.func_name, event.lineno))

        # Also compare call-entry locals vs return locals within the same function
        # to catch sign flips even when record_lines=False
        call_locals: Dict[str, Dict] = {}  # func_name -> locals at call time
        for event in self.events:
            if event.event_type == "call":
                call_locals[event.func_name] = event.locals_snapshot
            elif event.event_type == "return" and event.func_name in call_locals:
                entry = call_locals[event.func_name]
                for k, v in event.locals_snapshot.items():
                    if isinstance(v, (int, float)) and k in entry:
                        prev_v = entry[k]
                        if isinstance(prev_v, (int, float)):
                            if prev_v > 0 and v < 0:
                                # Check not already in num_history issues
                                key = ("sign_flip", k, event.func_name)
                                if not any(i["type"] == "sign_flip" and i["variable"] == k
                                           and i["func"] == event.func_name for i in issues):
                                    issues.append({
                                        "type": "sign_flip",
                                        "variable": k,
                                        "from_value": prev_v,
                                        "to_value": v,
                                        "event_id": event.event_id,
                                        "func": event.func_name,
                                        "lineno": event.lineno,
                                        "description": f"Variable `{k}` flipped from positive ({prev_v}) to negative ({v}) in {event.func_name}:{event.lineno}",
                                    })

        seen_sign_flips = set()
        for var, vals in num_history.items():
            for i in range(1, len(vals)):
                prev_id, prev_v, prev_fn, prev_ln = vals[i - 1]
                curr_id, curr_v, curr_fn, curr_ln = vals[i]
                # Sign flip (e.g., balance going negative)
                flip_key = (var, curr_fn)
                if prev_v > 0 and curr_v < 0 and flip_key not in seen_sign_flips:
                    seen_sign_flips.add(flip_key)
                    if not any(i["type"] == "sign_flip" and i["variable"] == var
                               and i["func"] == curr_fn for i in issues):
                        issues.append({
                            "type": "sign_flip",
                            "variable": var,
                            "from_value": prev_v,
                            "to_value": curr_v,
                            "event_id": curr_id,
                            "func": curr_fn,
                            "lineno": curr_ln,
                            "description": f"Variable `{var}` flipped from positive ({prev_v}) to negative ({curr_v}) in {curr_fn}:{curr_ln}",
                        })
                # Large sudden jump (> 10x)
                if prev_v != 0 and abs(curr_v / prev_v) > 10:
                    issues.append({
                        "type": "large_jump",
                        "variable": var,
                        "from_value": prev_v,
                        "to_value": curr_v,
                        "event_id": curr_id,
                        "func": curr_fn,
                        "lineno": curr_ln,
                        "description": f"Variable `{var}` jumped {prev_v} → {curr_v} (×{curr_v/prev_v:.1f}) in {curr_fn}:{curr_ln}",
                    })
        return issues

    def call_graph(self) -> Dict[str, List[str]]:
        """Build a call graph from the trace."""
        graph: Dict[str, List[str]] = {}
        stack = []
        for event in self.events:
            if event.event_type == "call":
                if stack:
                    caller = stack[-1]
                    if caller not in graph:
                        graph[caller] = []
                    if event.func_name not in graph[caller]:
                        graph[caller].append(event.func_name)
                stack.append(event.func_name)
            elif event.event_type == "return" and stack:
                stack.pop()
        return graph

    def hot_functions(self, top_n: int = 10) -> List[Dict]:
        """Return the most-called functions by call count."""
        counts: Dict[str, int] = {}
        for event in self.events:
            if event.event_type == "call":
                counts[event.func_name] = counts.get(event.func_name, 0) + 1
        return sorted(
            [{"func": k, "calls": v} for k, v in counts.items()],
            key=lambda x: x["calls"], reverse=True
        )[:top_n]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "label": self.label,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        data = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).write_text(data)
        return data

    @classmethod
    def from_json(cls, path: str) -> "TraceSession":
        data = json.loads(Path(path).read_text())
        session = cls(data["session_id"], data.get("label", ""))
        session.start_time = data["start_time"]
        session.end_time = data.get("end_time")
        for e in data.get("events", []):
            session.events.append(TraceEvent(**e))
        return session

    def timeline_summary(self, max_events: int = 50) -> str:
        """Return a human-readable timeline for LLM context."""
        lines = [f"=== Trace Session: {self.label or self.session_id} ===",
                 f"Duration: {self.duration_ms:.1f}ms | Events: {len(self.events)}",
                 ""]
        # Show key events (calls, exceptions, returns)
        key_events = [e for e in self.events
                      if e.event_type in ("call", "return", "exception")]
        sample = key_events[:max_events]
        for e in sample:
            lines.append(e.summary())
        if len(key_events) > max_events:
            lines.append(f"... and {len(key_events) - max_events} more events")

        exc = self.exceptions()
        if exc:
            lines.append(f"\n⚠️  {len(exc)} exception(s) recorded:")
            for e in exc[:5]:
                lines.append(f"  {e.exception_type}: {e.exception_msg} @ {Path(e.filename).name}:{e.lineno}")

        anomalies = self.anomalies()
        if anomalies:
            lines.append(f"\n🔍 {len(anomalies)} anomaly(ies) detected:")
            for a in anomalies[:5]:
                lines.append(f"  {a['description']}")

        return "\n".join(lines)


# ── Tracer ────────────────────────────────────────────────────────────────────

class Tracer:
    """
    Attaches to Python's trace machinery and records a TraceSession.

    Usage:
        tracer = Tracer(label="my_run", watch_modules=["myapp"])
        with tracer:
            my_function()
        session = tracer.session
    """

    def __init__(
        self,
        label: str = "",
        watch_modules: Optional[List[str]] = None,
        record_lines: bool = False,
        max_events: int = 100_000,
    ):
        self.label = label
        self.watch_modules = watch_modules  # None = watch everything
        self.record_lines = record_lines
        self.max_events = max_events
        self._event_counter = 0
        self._call_depth = 0
        self._call_start_times: Dict[int, float] = {}  # frame id → start time
        self.session = TraceSession(
            session_id=hashlib.md5(f"{label}{time.time()}".encode()).hexdigest()[:12],
            label=label,
        )

    def _should_trace(self, filename: str) -> bool:
        if not filename:
            return False
        # Allow inline code (e.g., exec() or -c) — filename is "<string>" etc.
        if filename.startswith("<"):
            return True
        if self.watch_modules is None:
            # Skip stdlib and site-packages, but only the tracer module itself
            skip = ["lib/python", "site-packages", "devmind/tracer/tracer.py"]
            return not any(s in filename for s in skip)
        return any(m in filename for m in self.watch_modules)

    def _trace_func(self, frame, event: str, arg):
        if self._event_counter >= self.max_events:
            return None

        filename = frame.f_code.co_filename
        if not self._should_trace(filename):
            return self._trace_func  # keep tracing but skip this frame

        func_name = frame.f_code.co_name
        lineno = frame.f_lineno
        module = frame.f_globals.get("__name__", "")

        # Snapshot locals (shallow copy to avoid mutation)
        try:
            locals_snap = {
                k: _safe_repr(v)
                for k, v in list(frame.f_locals.items())[:20]
            }
        except Exception:
            locals_snap = {}

        now = time.time()
        self._event_counter += 1

        if event == "call":
            self._call_depth += 1
            self._call_start_times[id(frame)] = now
            ev = TraceEvent(
                event_id=self._event_counter,
                event_type="call",
                timestamp=now,
                filename=filename,
                lineno=lineno,
                func_name=func_name,
                module=module,
                locals_snapshot=locals_snap,
                call_depth=self._call_depth,
            )
            self.session.add_event(ev)

        elif event == "return":
            elapsed = (now - self._call_start_times.pop(id(frame), now)) * 1000
            ev = TraceEvent(
                event_id=self._event_counter,
                event_type="return",
                timestamp=now,
                filename=filename,
                lineno=lineno,
                func_name=func_name,
                module=module,
                locals_snapshot=locals_snap,
                return_value=_safe_repr(arg),
                call_depth=self._call_depth,
                elapsed_ms=elapsed,
            )
            self.session.add_event(ev)
            self._call_depth = max(0, self._call_depth - 1)

        elif event == "exception":
            exc_type, exc_val, exc_tb = arg
            tb_str = "".join(traceback.format_tb(exc_tb))
            ev = TraceEvent(
                event_id=self._event_counter,
                event_type="exception",
                timestamp=now,
                filename=filename,
                lineno=lineno,
                func_name=func_name,
                module=module,
                locals_snapshot=locals_snap,
                exception_type=exc_type.__name__ if exc_type else "Unknown",
                exception_msg=str(exc_val),
                exception_tb=tb_str,
                call_depth=self._call_depth,
            )
            self.session.add_event(ev)

        elif event == "line" and self.record_lines:
            ev = TraceEvent(
                event_id=self._event_counter,
                event_type="line",
                timestamp=now,
                filename=filename,
                lineno=lineno,
                func_name=func_name,
                module=module,
                locals_snapshot=locals_snap,
                call_depth=self._call_depth,
            )
            self.session.add_event(ev)

        return self._trace_func

    def start(self):
        sys.settrace(self._trace_func)
        threading.settrace(self._trace_func)

    def stop(self):
        sys.settrace(None)
        threading.settrace(None)
        self.session.finish()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ── Decorator ─────────────────────────────────────────────────────────────────

def trace(label: str = "", watch_modules: Optional[List[str]] = None,
          record_lines: bool = False):
    """Decorator to trace a function and return (result, session)."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            t = Tracer(label=label or func.__name__,
                       watch_modules=watch_modules,
                       record_lines=record_lines)
            t.start()
            try:
                result = func(*args, **kwargs)
            finally:
                t.stop()
            return result, t.session
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
