"""
DevMind Infrastructure Analyzer
================================
Ingests logs, metrics, and deploy events from multiple sources
and builds a structured timeline for LLM-powered root cause analysis.

Supports:
- Structured JSON logs (any format)
- Unstructured text logs (with regex parsing)
- Simulated metrics (latency, error rate, throughput)
- Deploy events (git commits, version bumps)
- Anomaly detection (error rate spikes, latency p99 jumps)
"""

import re
import json
import time
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict


# ── Log entry model ───────────────────────────────────────────────────────────

@dataclass
class LogEntry:
    entry_id: int
    timestamp: float
    level: str          # DEBUG | INFO | WARN | ERROR | FATAL
    service: str
    message: str
    fields: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def is_error(self) -> bool:
        return self.level in ("ERROR", "FATAL")

    def summary(self) -> str:
        ts = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
        svc = f"[{self.service}]" if self.service else ""
        lat = f" ({self.latency_ms:.0f}ms)" if self.latency_ms else ""
        return f"[{ts}] {self.level:5s} {svc} {self.message[:120]}{lat}"


@dataclass
class MetricPoint:
    timestamp: float
    service: str
    metric: str         # latency_p99 | error_rate | throughput | cpu | memory
    value: float
    unit: str = ""


@dataclass
class DeployEvent:
    timestamp: float
    service: str
    version: str
    commit: str
    author: str
    description: str


# ── Log parser ────────────────────────────────────────────────────────────────

# Common log patterns
_PATTERNS = [
    # JSON log
    ("json", re.compile(r"^\s*\{")),
    # Apache/Nginx combined log
    ("apache", re.compile(
        r'(?P<ip>[\d.]+) .+ \[(?P<time>[^\]]+)\] "(?P<method>\w+) (?P<path>[^ ]+)[^"]*" (?P<status>\d+) (?P<size>\d+)'
    )),
    # Python logging format
    ("python", re.compile(
        r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.\d]*) (?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) (?P<logger>[^ ]+) (?P<msg>.*)"
    )),
    # Generic level + message
    ("generic", re.compile(
        r"(?P<time>\d{2}:\d{2}:\d{2}[.\d]*)\s+(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL)\s+(?P<msg>.*)"
    )),
    # Kubernetes/Docker log prefix
    ("k8s", re.compile(
        r"(?P<time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.\d]*Z?)\s+(?P<level>debug|info|warn|error|fatal)\s+(?P<msg>.*)"
    )),
]

_LEVEL_MAP = {
    "WARNING": "WARN", "CRITICAL": "FATAL",
    "debug": "DEBUG", "info": "INFO", "warn": "WARN",
    "error": "ERROR", "fatal": "FATAL",
}


def _parse_timestamp(ts_str: str) -> float:
    """Try multiple timestamp formats."""
    fmts = [
        "%Y-%m-%d %H:%M:%S,%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d/%b/%Y:%H:%M:%S",
        "%H:%M:%S.%f",
        "%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(ts_str.strip(), fmt)
            if dt.year == 1900:
                dt = dt.replace(year=datetime.now().year)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return time.time()


def parse_log_line(line: str, service: str = "unknown", entry_id: int = 0) -> Optional[LogEntry]:
    """Parse a single log line into a LogEntry."""
    line = line.strip()
    if not line:
        return None

    # Try JSON
    if line.startswith("{"):
        try:
            data = json.loads(line)
            ts = data.get("timestamp") or data.get("time") or data.get("ts") or time.time()
            if isinstance(ts, str):
                ts = _parse_timestamp(ts)
            level = str(data.get("level", data.get("severity", "INFO"))).upper()
            level = _LEVEL_MAP.get(level, level)
            msg = data.get("message", data.get("msg", data.get("text", str(data))))
            svc = data.get("service", data.get("app", service))
            return LogEntry(
                entry_id=entry_id,
                timestamp=float(ts),
                level=level,
                service=svc,
                message=str(msg),
                fields={k: v for k, v in data.items()
                        if k not in ("timestamp", "time", "ts", "level", "severity", "message", "msg", "service", "app")},
                trace_id=data.get("trace_id") or data.get("traceId"),
                span_id=data.get("span_id") or data.get("spanId"),
                error_type=data.get("error_type") or data.get("errorType"),
                latency_ms=data.get("latency_ms") or data.get("duration_ms"),
            )
        except json.JSONDecodeError:
            pass

    # Try Python logging
    m = _PATTERNS[2][1].match(line)
    if m:
        level = m.group("level").upper()
        level = _LEVEL_MAP.get(level, level)
        return LogEntry(
            entry_id=entry_id,
            timestamp=_parse_timestamp(m.group("time")),
            level=level,
            service=m.group("logger").split(".")[-1],
            message=m.group("msg"),
        )

    # Try k8s
    m = _PATTERNS[4][1].match(line)
    if m:
        level = m.group("level").upper()
        level = _LEVEL_MAP.get(level, level)
        return LogEntry(
            entry_id=entry_id,
            timestamp=_parse_timestamp(m.group("time")),
            level=level,
            service=service,
            message=m.group("msg"),
        )

    # Try generic
    m = _PATTERNS[3][1].match(line)
    if m:
        level = m.group("level").upper()
        level = _LEVEL_MAP.get(level, level)
        return LogEntry(
            entry_id=entry_id,
            timestamp=_parse_timestamp(m.group("time")),
            level=level,
            service=service,
            message=m.group("msg"),
        )

    # Fallback: treat entire line as INFO message
    return LogEntry(
        entry_id=entry_id,
        timestamp=time.time(),
        level="INFO",
        service=service,
        message=line[:500],
    )


# ── Infrastructure session ────────────────────────────────────────────────────

class InfraSession:
    """
    A collection of logs, metrics, and deploy events for a time window.
    Provides anomaly detection and structured summaries for LLM analysis.
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.logs: List[LogEntry] = []
        self.metrics: List[MetricPoint] = []
        self.deploys: List[DeployEvent] = []
        self._entry_counter = 0

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_log_text(self, text: str, service: str = "unknown"):
        """Parse and ingest multi-line log text."""
        for line in text.splitlines():
            entry = parse_log_line(line, service=service, entry_id=self._entry_counter)
            if entry:
                self.logs.append(entry)
                self._entry_counter += 1

    def ingest_log_file(self, path: str, service: str = ""):
        """Parse and ingest a log file."""
        svc = service or Path(path).stem
        self.ingest_log_text(Path(path).read_text(errors="replace"), service=svc)

    def add_metric(self, service: str, metric: str, value: float,
                   timestamp: Optional[float] = None, unit: str = ""):
        self.metrics.append(MetricPoint(
            timestamp=timestamp or time.time(),
            service=service, metric=metric, value=value, unit=unit,
        ))

    def add_deploy(self, service: str, version: str, commit: str = "",
                   author: str = "", description: str = "",
                   timestamp: Optional[float] = None):
        self.deploys.append(DeployEvent(
            timestamp=timestamp or time.time(),
            service=service, version=version, commit=commit,
            author=author, description=description,
        ))

    # ── Analysis ───────────────────────────────────────────────────────────────

    def error_rate(self, service: Optional[str] = None,
                   window_seconds: float = 300) -> float:
        """Error rate (0–1) for the last N seconds."""
        cutoff = time.time() - window_seconds
        entries = [l for l in self.logs if l.timestamp >= cutoff]
        if service:
            entries = [l for l in entries if l.service == service]
        if not entries:
            return 0.0
        errors = sum(1 for l in entries if l.is_error())
        return errors / len(entries)

    def error_clusters(self) -> List[Dict]:
        """Group errors by message pattern to find recurring issues."""
        clusters: Dict[str, List[LogEntry]] = defaultdict(list)
        for entry in self.logs:
            if entry.is_error():
                # Normalize: strip numbers and UUIDs for clustering
                key = re.sub(r"\b[0-9a-f-]{8,}\b", "<id>",
                             re.sub(r"\d+", "<n>", entry.message))[:80]
                clusters[key].append(entry)
        return sorted(
            [{"pattern": k, "count": len(v), "examples": [e.message for e in v[:3]],
              "first_seen": min(e.timestamp for e in v),
              "last_seen": max(e.timestamp for e in v)}
             for k, v in clusters.items()],
            key=lambda x: x["count"], reverse=True
        )

    def latency_anomalies(self) -> List[Dict]:
        """Find services with sudden latency spikes."""
        anomalies = []
        # Group latency metrics by service
        by_service: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for m in self.metrics:
            if "latency" in m.metric:
                by_service[m.service].append((m.timestamp, m.value))

        for service, points in by_service.items():
            if len(points) < 5:
                continue
            points.sort(key=lambda x: x[0])
            values = [p[1] for p in points]
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0
            if stdev == 0:
                continue
            for i, (ts, val) in enumerate(points):
                z_score = (val - mean) / stdev
                if z_score > 2.5:
                    anomalies.append({
                        "service": service,
                        "metric": "latency",
                        "timestamp": ts,
                        "value": val,
                        "mean": mean,
                        "z_score": z_score,
                        "description": f"{service} latency spike: {val:.0f}ms (mean={mean:.0f}ms, z={z_score:.1f}σ)",
                    })
        return anomalies

    def deploy_correlations(self) -> List[Dict]:
        """Find errors that started after a deploy (within 10 minutes)."""
        correlations = []
        for deploy in self.deploys:
            window_start = deploy.timestamp
            window_end = deploy.timestamp + 600  # 10 minutes
            post_deploy_errors = [
                l for l in self.logs
                if l.is_error() and window_start <= l.timestamp <= window_end
                and (l.service == deploy.service or not l.service)
            ]
            if post_deploy_errors:
                correlations.append({
                    "deploy": {
                        "service": deploy.service,
                        "version": deploy.version,
                        "commit": deploy.commit,
                        "timestamp": deploy.timestamp,
                    },
                    "errors_within_10min": len(post_deploy_errors),
                    "error_samples": [e.message for e in post_deploy_errors[:3]],
                    "description": f"Deploy {deploy.service} v{deploy.version} followed by {len(post_deploy_errors)} errors",
                })
        return correlations

    def timeline_summary(self, max_errors: int = 20) -> str:
        """Produce a structured summary for LLM context."""
        lines = [f"=== Infrastructure Session: {self.label} ===",
                 f"Logs: {len(self.logs)} | Metrics: {len(self.metrics)} | Deploys: {len(self.deploys)}",
                 ""]

        # Error summary
        errors = [l for l in self.logs if l.is_error()]
        if errors:
            lines.append(f"❌ {len(errors)} ERROR/FATAL log entries:")
            for e in errors[:max_errors]:
                lines.append(f"  {e.summary()}")
            if len(errors) > max_errors:
                lines.append(f"  ... and {len(errors) - max_errors} more")
            lines.append("")

        # Error clusters
        clusters = self.error_clusters()
        if clusters:
            lines.append(f"🔁 Top recurring error patterns:")
            for c in clusters[:5]:
                lines.append(f"  [{c['count']}x] {c['pattern']}")
            lines.append("")

        # Latency anomalies
        lat_anomalies = self.latency_anomalies()
        if lat_anomalies:
            lines.append(f"⚡ Latency anomalies:")
            for a in lat_anomalies[:5]:
                lines.append(f"  {a['description']}")
            lines.append("")

        # Deploy correlations
        deploy_corr = self.deploy_correlations()
        if deploy_corr:
            lines.append(f"🚀 Deploy correlations:")
            for d in deploy_corr[:5]:
                lines.append(f"  {d['description']}")
                for sample in d["error_samples"]:
                    lines.append(f"    → {sample[:100]}")
            lines.append("")

        # Recent log tail
        recent = sorted(self.logs, key=lambda l: l.timestamp)[-20:]
        if recent:
            lines.append("📋 Recent log entries:")
            for l in recent:
                lines.append(f"  {l.summary()}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "logs": [l.to_dict() for l in self.logs],
            "metrics": [asdict(m) for m in self.metrics],
            "deploys": [asdict(d) for d in self.deploys],
        }
