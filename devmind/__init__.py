"""
DevMind — AI-Powered Debugging Copilot
=======================================
Combines a Python execution Time Machine with an LLM-powered
infrastructure analysis engine to help developers find root causes fast.

Quick start:
    from devmind import Tracer, InfraSession, MockEngine

    # Trace a function
    with Tracer(label="my_run") as t:
        my_function()

    # Ask a question
    engine = MockEngine()
    answer = engine.analyze_trace(t.session)
    print(answer.format())
"""

from devmind.tracer.tracer import Tracer, TraceSession, TraceEvent, trace
from devmind.infra.analyzer import InfraSession, LogEntry
from devmind.llm.engine import MockEngine, DebugAnswer, RootCause

__version__ = "1.0.0"
__all__ = [
    "Tracer", "TraceSession", "TraceEvent", "trace",
    "InfraSession", "LogEntry",
    "MockEngine", "DebugAnswer", "RootCause",
]
