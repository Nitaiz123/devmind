"""
DevMind LLM Query Engine
=========================
Provides natural language querying over TraceSession and InfraSession data.

The engine:
1. Builds a rich context window from the session data
2. Sends a structured prompt to the LLM
3. Returns a structured RootCauseAnalysis or DebugAnswer

Supports:
- OpenAI-compatible APIs (GPT-4, local Ollama, etc.)
- Streaming responses
- Multi-turn conversation (session memory)
- Structured output (JSON mode)
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ── Response models ───────────────────────────────────────────────────────────

@dataclass
class RootCause:
    title: str
    confidence: float          # 0.0 – 1.0
    description: str
    evidence: List[str]        # log lines / trace events that support this
    affected_component: str
    suggested_fix: str
    fix_code_snippet: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "confidence": self.confidence,
            "description": self.description,
            "evidence": self.evidence,
            "affected_component": self.affected_component,
            "suggested_fix": self.suggested_fix,
            "fix_code_snippet": self.fix_code_snippet,
        }


@dataclass
class DebugAnswer:
    question: str
    answer: str
    root_causes: List[RootCause] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    model_used: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "follow_up_questions": self.follow_up_questions,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
        }

    def format(self) -> str:
        """Pretty-print the answer for CLI output."""
        lines = [
            f"\n{'='*60}",
            f"🤖 DevMind Analysis",
            f"{'='*60}",
            f"\n{self.answer}",
        ]
        if self.root_causes:
            lines.append(f"\n{'─'*60}")
            lines.append("📍 Root Causes:")
            for i, rc in enumerate(self.root_causes, 1):
                conf_bar = "█" * int(rc.confidence * 10) + "░" * (10 - int(rc.confidence * 10))
                lines.append(f"\n  {i}. {rc.title} [{conf_bar}] {rc.confidence*100:.0f}%")
                lines.append(f"     {rc.description}")
                lines.append(f"     📌 Affected: {rc.affected_component}")
                lines.append(f"     💡 Fix: {rc.suggested_fix}")
                if rc.fix_code_snippet:
                    lines.append(f"     ```\n{rc.fix_code_snippet}\n     ```")
                if rc.evidence:
                    lines.append(f"     🔍 Evidence:")
                    for ev in rc.evidence[:3]:
                        lines.append(f"       • {ev}")
        if self.follow_up_questions:
            lines.append(f"\n{'─'*60}")
            lines.append("💬 You might also ask:")
            for q in self.follow_up_questions:
                lines.append(f"  • {q}")
        lines.append(f"\n{'─'*60}")
        lines.append(f"Model: {self.model_used} | Tokens: {self.tokens_used} | {self.latency_ms:.0f}ms")
        return "\n".join(lines)


# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are DevMind, an expert AI debugging assistant for software engineers.
You have access to execution traces (function calls, variable states, exceptions) and
infrastructure data (logs, metrics, deploy events).

Your job is to:
1. Answer developer questions about what went wrong and why
2. Identify root causes with supporting evidence from the data
3. Suggest concrete fixes with code snippets when possible
4. Highlight anomalies, patterns, and correlations the developer may have missed

Always be specific — reference exact function names, line numbers, variable values,
and log messages from the provided data. Never make up data that isn't in the context.

Respond in the following JSON format:
{
  "answer": "<clear, direct answer to the question>",
  "root_causes": [
    {
      "title": "<short title>",
      "confidence": <0.0-1.0>,
      "description": "<detailed explanation>",
      "evidence": ["<log line or trace event>", ...],
      "affected_component": "<function/service/module name>",
      "suggested_fix": "<concrete fix description>",
      "fix_code_snippet": "<optional code snippet>"
    }
  ],
  "follow_up_questions": ["<question 1>", "<question 2>", "<question 3>"]
}"""

_CONTEXT_TEMPLATE = """
## Execution Trace
{trace_summary}

## Infrastructure / Logs
{infra_summary}

## Developer Question
{question}
"""


# ── LLM Engine ────────────────────────────────────────────────────────────────

class DevMindEngine:
    """
    The core LLM-powered analysis engine.

    Usage:
        engine = DevMindEngine()
        answer = engine.ask(
            question="Why did user.balance go negative?",
            trace_session=session,
            infra_session=infra,
        )
        print(answer.format())
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_context_tokens: int = 8000,
    ):
        self.model = model
        self.max_context_tokens = max_context_tokens
        self._history: List[Dict] = []

        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package required: pip install openai")

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_API_BASE"),
        )

    def ask(
        self,
        question: str,
        trace_session=None,
        infra_session=None,
        extra_context: str = "",
        stream: bool = False,
    ) -> DebugAnswer:
        """Ask a natural language question about the session data."""
        start = time.time()

        # Build context
        trace_summary = trace_session.timeline_summary() if trace_session else "No execution trace provided."
        infra_summary = infra_session.timeline_summary() if infra_session else "No infrastructure data provided."

        context = _CONTEXT_TEMPLATE.format(
            trace_summary=trace_summary,
            infra_summary=infra_summary,
            question=question,
        )
        if extra_context:
            context = extra_context + "\n\n" + context

        # Truncate context if too long (rough token estimate: 4 chars ≈ 1 token)
        max_chars = self.max_context_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...[truncated]"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": context},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
        except Exception as e:
            # Graceful fallback
            raw = json.dumps({
                "answer": f"LLM call failed: {e}. Please check your API key and model configuration.",
                "root_causes": [],
                "follow_up_questions": [],
            })
            tokens = 0

        latency_ms = (time.time() - start) * 1000

        # Parse response
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"answer": raw, "root_causes": [], "follow_up_questions": []}

        root_causes = []
        for rc_data in data.get("root_causes", []):
            root_causes.append(RootCause(
                title=rc_data.get("title", "Unknown"),
                confidence=float(rc_data.get("confidence", 0.5)),
                description=rc_data.get("description", ""),
                evidence=rc_data.get("evidence", []),
                affected_component=rc_data.get("affected_component", ""),
                suggested_fix=rc_data.get("suggested_fix", ""),
                fix_code_snippet=rc_data.get("fix_code_snippet"),
            ))

        answer = DebugAnswer(
            question=question,
            answer=data.get("answer", ""),
            root_causes=root_causes,
            follow_up_questions=data.get("follow_up_questions", []),
            model_used=self.model,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )

        # Add to conversation history
        self._history.append({"role": "user", "content": context})
        self._history.append({"role": "assistant", "content": raw})
        # Keep history bounded
        if len(self._history) > 20:
            self._history = self._history[-20:]

        return answer

    def ask_followup(self, question: str) -> DebugAnswer:
        """Ask a follow-up question using the existing conversation history."""
        return self.ask(question)

    def clear_history(self):
        """Clear conversation history."""
        self._history = []

    def analyze_trace(self, trace_session) -> DebugAnswer:
        """Auto-analyze a trace session without a specific question."""
        question = (
            "Analyze this execution trace. Identify any bugs, exceptions, "
            "performance issues, or suspicious variable state changes. "
            "Provide a root cause analysis with specific evidence and fixes."
        )
        return self.ask(question, trace_session=trace_session)

    def analyze_infra(self, infra_session) -> DebugAnswer:
        """Auto-analyze infrastructure logs and metrics."""
        question = (
            "Analyze these infrastructure logs and metrics. "
            "Identify any errors, latency spikes, deploy-correlated incidents, "
            "or recurring failure patterns. Provide a prioritized list of issues with fixes."
        )
        return self.ask(question, infra_session=infra_session)

    def diagnose_incident(self, trace_session=None, infra_session=None,
                          symptom: str = "") -> DebugAnswer:
        """Full incident diagnosis combining trace + infra data."""
        question = (
            f"We have a production incident. Symptom: {symptom or 'Unknown issue'}. "
            "Using both the execution trace and infrastructure data, "
            "identify the root cause, the blast radius, and the fastest path to resolution. "
            "Be specific about which service, function, or code change caused this."
        )
        return self.ask(question, trace_session=trace_session, infra_session=infra_session)


# ── Offline / mock engine for testing ────────────────────────────────────────

class MockEngine:
    """
    A mock engine that returns canned responses for testing without an API key.
    Demonstrates the same interface as DevMindEngine.
    """

    def ask(self, question: str, trace_session=None, infra_session=None,
            extra_context: str = "", stream: bool = False) -> DebugAnswer:
        # Build a simple heuristic answer from the session data
        root_causes = []
        answer_parts = [f"Analyzing your question: '{question}'"]

        if trace_session:
            exceptions = trace_session.exceptions()
            anomalies = trace_session.anomalies()
            if exceptions:
                exc = exceptions[0]
                root_causes.append(RootCause(
                    title=f"Unhandled {exc.exception_type}",
                    confidence=0.92,
                    description=f"Exception '{exc.exception_type}: {exc.exception_msg}' raised in {exc.func_name} at line {exc.lineno}",
                    evidence=[exc.summary()],
                    affected_component=exc.func_name,
                    suggested_fix=f"Add try/except around the call to {exc.func_name} and handle {exc.exception_type}",
                    fix_code_snippet=f"try:\n    result = {exc.func_name}(...)\nexcept {exc.exception_type} as e:\n    logger.error(f'Failed: {{e}}')\n    # handle gracefully",
                ))
                answer_parts.append(f"Found {len(exceptions)} exception(s). Primary: {exc.exception_type} in {exc.func_name}.")
            if anomalies:
                a = anomalies[0]
                root_causes.append(RootCause(
                    title=f"Variable anomaly: {a['variable']}",
                    confidence=0.78,
                    description=a["description"],
                    evidence=[a["description"]],
                    affected_component=a["func"],
                    suggested_fix=f"Add validation for `{a['variable']}` before it is used in {a['func']}",
                ))
                answer_parts.append(f"Detected {len(anomalies)} variable anomaly(ies).")

        if infra_session:
            clusters = infra_session.error_clusters()
            deploy_corr = infra_session.deploy_correlations()
            if deploy_corr:
                d = deploy_corr[0]
                root_causes.append(RootCause(
                    title=f"Deploy-correlated errors: {d['deploy']['service']} v{d['deploy']['version']}",
                    confidence=0.85,
                    description=d["description"],
                    evidence=d["error_samples"],
                    affected_component=d["deploy"]["service"],
                    suggested_fix=f"Roll back {d['deploy']['service']} to the previous version and investigate commit {d['deploy']['commit']}",
                ))
                answer_parts.append(f"Deploy correlation detected: {d['description']}")
            elif clusters:
                c = clusters[0]
                root_causes.append(RootCause(
                    title=f"Recurring error: {c['pattern'][:50]}",
                    confidence=0.70,
                    description=f"Error pattern '{c['pattern']}' occurred {c['count']} times",
                    evidence=c["examples"],
                    affected_component="unknown",
                    suggested_fix="Investigate the root cause of this recurring error and add proper error handling",
                ))
                answer_parts.append(f"Found {len(clusters)} error cluster(s). Top: {c['pattern'][:60]}")

        if not root_causes:
            answer_parts.append("No obvious issues detected in the provided data. The system appears healthy.")

        return DebugAnswer(
            question=question,
            answer=" ".join(answer_parts),
            root_causes=root_causes,
            follow_up_questions=[
                "What was the state of the database at the time of the error?",
                "Were there any recent configuration changes?",
                "Is this error reproducible in a local environment?",
            ],
            model_used="devmind-mock",
            tokens_used=0,
            latency_ms=5.0,
        )

    def analyze_trace(self, trace_session) -> DebugAnswer:
        return self.ask("Analyze this trace", trace_session=trace_session)

    def analyze_infra(self, infra_session) -> DebugAnswer:
        return self.ask("Analyze infrastructure", infra_session=infra_session)

    def diagnose_incident(self, trace_session=None, infra_session=None,
                          symptom: str = "") -> DebugAnswer:
        return self.ask(f"Diagnose incident: {symptom}",
                        trace_session=trace_session, infra_session=infra_session)

    def ask_followup(self, question: str) -> DebugAnswer:
        return self.ask(question)

    def clear_history(self):
        pass
