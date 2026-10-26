"""Strands tools: the agent's hands. Each tool call is trace-logged.

Signatures are agent-friendly (all args have defaults; the runner stashes the
current clip path on the function objects so the VLM agent only needs to pass
label/confidence/score, which it always knows).
"""
from __future__ import annotations
import os

try:
    from strands import tool
except ImportError:  # mock/CI without strands installed: no-op decorator
    def tool(fn=None, **kw):
        return fn if callable(fn) else (lambda f: f)

from .bedrock import confirm_posture
from .store import store_clip


@tool
def confirm_posture_tool(trunk_flexion_deg: float, score: float) -> dict:
    """Ask the Bedrock VLM to confirm or reject the OpenCV risk flag.

    The runner attaches the event frames + full OpenCV angles; the agent only
    passes the two numbers from its prompt.
    """
    frames = getattr(confirm_posture_tool, "_frames", [])
    angles = getattr(confirm_posture_tool, "_angles",
                     {"trunk_flexion_deg": trunk_flexion_deg})
    return confirm_posture(frames, angles, score,
                           None if os.getenv("ERGO_MOCK_BEDROCK", "1") == "1"
                           else os.getenv("BEDROCK_MODEL_ID"))


@tool
def store_clip_tool(label: str, confidence: float, score: float,
                    clip_path: str = "") -> dict:
    """Persist a confirmed-dangerous clip to S3 with metadata."""
    return store_clip(clip_path or getattr(store_clip_tool, "_clip", ""),
                      {"label": label, "vlm_confidence": confidence,
                       "opencv_score": score, "decision": "STORE"})


@tool
def request_review_tool(label: str, confidence: float, score: float,
                        clip_path: str = "") -> dict:
    """Escalate an uncertain case to the human review queue."""
    import json, time
    os.makedirs("data/review", exist_ok=True)
    rid = f"rev_{int(time.time() * 1000)}"
    rec = {"id": rid, "clip": clip_path or getattr(request_review_tool, "_clip", ""),
           "label": label, "confidence": confidence, "score": score,
           "status": "pending"}
    with open(f"data/review/{rid}.json", "w") as f:
        json.dump(rec, f, indent=2)
    return rec


@tool
def discard_tool(score: float, label: str) -> dict:
    """Discard a VLM-rejected false alarm; log the miss for threshold tuning."""
    import json, time
    os.makedirs("data", exist_ok=True)
    row = {"ts": time.time(), "score": score, "label": label, "outcome": "DISCARD"}
    with open("data/misses.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row
