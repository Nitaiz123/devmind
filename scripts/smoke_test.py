"""
DevMind smoke test — runs without any API key.
Used by the CI demo job.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devmind import Tracer, MockEngine


def buggy():
    balance = 100
    balance -= 200
    return balance


with Tracer(label="ci_test", record_lines=True) as t:
    buggy()

engine = MockEngine()
answer = engine.analyze_trace(t.session)
assert len(answer.root_causes) > 0, f"Expected root causes, got: {answer.answer}"
print("Demo smoke test passed:", answer.answer[:80])
print("Root causes found:", len(answer.root_causes))
