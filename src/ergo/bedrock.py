"""Bedrock multimodal second look. Real path uses converse with JPEG frames;
mock path returns a deterministic verdict from the OpenCV score so the
demo/trace runs with zero AWS cost (ERGO_MOCK_BEDROCK=1)."""
from __future__ import annotations
import base64, os, cv2

MOCK = os.getenv("ERGO_MOCK_BEDROCK", "1") == "1"

PROMPT = ("You are a warehouse ergonomics reviewer. Given CCTV frames and "
  "OpenCV joint angles, classify the lifting posture as one of: "
  "safe_lift | bending_risk | overreaching_risk | twisting_risk | false_alarm. "
  "Return JSON: {label, confidence 0..1, rationale<=25 words, needs_human bool}.")

def _jpg_b64(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode()

def confirm_posture(frames: list, angles: dict, score: float,
                    model_id: str | None = None) -> dict:
    """Called by the Strands tool; real Bedrock path below, mock when env set."""
    if MOCK or model_id is None:
        s = score
        if s >= 0.8:   label, conf = "bending_risk", 0.91
        elif s >= 0.65: label, conf = "bending_risk", 0.72
        elif s >= 0.45: label, conf = "overreaching_risk", 0.55
        else:           label, conf = "false_alarm", 0.88
        return {"label": label, "confidence": conf, "needs_human": 0.45 <= s < 0.65,
                "rationale": f"mock: trunk {angles.get('trunk_flexion_deg',0):.0f}deg score {s}",
                "mock": True}
    import boto3  # real path: Bedrock converse, multimodal
    br = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    content = [{"text": f"{PROMPT}\nOpenCV angles: {angles}\nOpenCV score: {score}"}]
    for f in frames[:5]:
        content.append({"image": {"format": "jpeg", "source": {"bytes": base64.b64decode(_jpg_b64(f))}}})
    resp = br.converse(modelId=model_id, messages=[{"role": "user", "content": content}],
                       inferenceConfig={"maxTokens": 300, "temperature": 0.1})
    text = resp["output"]["message"]["content"][0].get("text", "{}")
    import json
    try: out = json.loads(text[text.index("{"):text.rindex("}")+1])
    except Exception: out = {"label": "bending_risk", "confidence": 0.5, "needs_human": True,
                             "rationale": "unparseable model output, escalating"}
    out["mock"] = False
    return out
