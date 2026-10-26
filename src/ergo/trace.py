"""Append-only JSONL trace. This file IS the Agentic Vision award evidence:
each row links OpenCV output -> agent reasoning -> action taken."""
from __future__ import annotations
import json, os, time

class Tracer:
    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("ERGO_TRACE_PATH", "data/trace.jsonl")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
    def log(self, event: str, **fields) -> dict:
        row = {"ts": time.time(), "event": event, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row
