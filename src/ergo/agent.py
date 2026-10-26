"""Strands agent: perception -> decision -> action loop.

Real path (Bedrock creds): a Strands Agent built on BedrockModel owns the
decision. OpenCV 5 numbers go into its prompt; it must call
confirm_posture_tool first, then exactly one of store / review / discard.
The wrapper reads back which action-tool fired from agent.messages and logs
the same trace events as the mock path, so both modes produce identical
award-trace evidence.

Fallback: if the Strands/bedrock call fails (no creds, throttle), the wrapper
re-runs the deterministic safety policy with the last VLM verdict (or the
mock verdict) so the pipeline never silently drops an event.
"""
from __future__ import annotations
import json
import os

from . import tools as T
from .trace import Tracer

SYSTEM = (
    "You are ergo-safety, a warehouse ergonomics agent. "
    "You receive OpenCV 5 pose angles + a risk score for a CCTV frame window. "
    "ALWAYS call confirm_posture_tool first with the given score and trunk angle. "
    "Then follow this policy exactly:\n"
    "- confidence >= 0.75 and label != false_alarm -> store_clip_tool\n"
    "- needs_human true, or 0.45 <= confidence < 0.75 -> request_review_tool\n"
    "- otherwise -> discard_tool.\n"
    "Reply with one short line stating the tool you chose and why."
)

ACTION_TOOLS = {"store_clip_tool": "STORE",
                "request_review_tool": "HUMAN_REVIEW",
                "discard_tool": "DISCARD"}


def make_agent(mock: bool | None = None):
    mock = os.getenv("ERGO_MOCK_BEDROCK", "1") == "1" if mock is None else mock
    if mock:
        return _AgentRunner(None)
    try:
        from strands import Agent
        from strands.models import BedrockModel
        model = BedrockModel(
            model_id=os.getenv("BEDROCK_MODEL_ID",
                               "anthropic.claude-3-5-sonnet-20241022-v2:0"))
        agent = Agent(model=model, system_prompt=SYSTEM,
                      tools=[T.confirm_posture_tool, T.store_clip_tool,
                             T.request_review_tool, T.discard_tool])
        return _AgentRunner(agent)
    except Exception as e:
        print(f"ergo: Strands/Bedrock unavailable ({e}) -> deterministic policy")
        return _AgentRunner(None)


class _AgentRunner:
    """Uniform .run() for live-Strands and deterministic-policy modes."""
    def __init__(self, agent):
        self.agent = agent
        self.tracer = Tracer()

    def run(self, opencv: dict, frames: list, clip_path: str) -> dict:
        score = opencv["score"]
        T.confirm_posture_tool._frames = (frames or [])[-5:]
        T.confirm_posture_tool._angles = opencv["angles"]
        T.store_clip_tool._clip = clip_path
        T.request_review_tool._clip = clip_path
        self.tracer.log("agent_perception", opencv_score=score,
                        trunk_deg=round(opencv["angles"].get("trunk_flexion_deg", 0), 1))

        verdict, decision, result = None, None, None
        if self.agent is not None:
            try:
                resp = self.agent(
                    f"OpenCV 5 window: risk score {score} (trigger threshold "
                    f"{os.getenv('ERGO_THRESHOLD', '0.65')}), angles "
                    f"{json.dumps(opencv['angles'])}.")
                verdict, decision, result = self._read_tool_outcome(str(resp))
            except Exception as e:
                self.tracer.log("agent_fallback", reason=str(e)[:200])
                self.agent = None  # fall through to deterministic policy
        if decision is None:
            verdict, decision, result = self._policy(score, opencv, clip_path)
        self.tracer.log("agent_decision", decision=decision, vlm=verdict,
                        tool_result=json.dumps(result, default=str)[:400])
        return {"decision": decision, "vlm": verdict, "result": result,
                "trace_note": f"OpenCV {score} -> VLM {verdict} -> {decision}"}

    def _read_tool_outcome(self, agent_text: str):
        """Recover (verdict, decision, result) from the tools the agent fired."""
        verdict, decision, result = None, None, None
        for msg in reversed(getattr(self.agent, "messages", []) or []):
            for block in (msg.get("content") or []):
                tu = block.get("toolUse") or {}
                name = tu.get("name")
                if name == "confirm_posture_tool":
                    for m2 in getattr(self.agent, "messages", []):
                        for b2 in (m2.get("content") or []):
                            tr = b2.get("toolResult") or {}
                            if tr.get("toolUseId") == tu.get("toolUseId"):
                                content = tr.get("content") or {}
                                txt = (content.get("text") or "")
                                try:
                                    verdict = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
                                except Exception:
                                    pass
                elif name in ACTION_TOOLS:
                    decision = ACTION_TOOLS[name]
                    result = {"tool": name, "input": tu.get("input", {})}
        return verdict or {}, decision, result or {"note": agent_text[:200]}

    def _policy(self, score: float, opencv: dict, clip_path: str):
        """Deterministic policy: same rules as SYSTEM, executed in-process."""
        verdict = T.confirm_posture_tool(
            score=score,
            trunk_flexion_deg=opencv["angles"].get("trunk_flexion_deg", 0.0))
        verdict = dict(verdict)
        label, conf = verdict.get("label", "?"), float(verdict.get("confidence", 0))
        self.tracer.log("agent_tool_call", tool="confirm_posture_tool", result=verdict)
        if conf >= 0.75 and label != "false_alarm":
            return verdict, "STORE", T.store_clip_tool(
                clip_path=clip_path or "data/clips/demo.mp4", label=label,
                confidence=conf, score=score)
        if verdict.get("needs_human") or 0.45 <= conf < 0.75:
            return verdict, "HUMAN_REVIEW", T.request_review_tool(
                clip_path=clip_path or "data/clips/demo.mp4", label=label,
                confidence=conf, score=score)
        return verdict, "DISCARD", T.discard_tool(score=score, label=label)
