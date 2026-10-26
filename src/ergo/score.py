"""Posture risk score 0..1 (0 = clean, 1 = worst-case lift).

Weighted trunk/reach/legs/hand-height terms with visibility downweighting and
an RNLE-informed boost: H in [25,63]cm is nominal, H>63cm or hands far from
body sharply reduces the Recommended Weight Limit, so it raises the score.
The threshold is a *gate*, not a verdict — the VLM makes the call.
"""
from __future__ import annotations
import time

RNLE_H_SAFE, RNLE_H_MAX = 25.0, 63.0

def risk_score(a: dict) -> dict:
    trunk = min(a["trunk_flexion_deg"] / 90.0, 1.0)
    reach = min(a["H_norm"] / 0.55, 1.0)
    legs = 1.0 - min(a["knee_flexion_deg"] / 40.0, 1.0)   # straight-leg lift
    low = min(max((0.55 - a["V_norm"]) / 0.35, 0.0), 1.0)  # hands near the floor
    score = 0.42 * trunk + 0.24 * reach + 0.14 * legs + 0.20 * low

    h_cm = a.get("H_cm")
    if h_cm is not None:                                   # physical distance term
        over = min(max((h_cm - RNLE_H_SAFE) / (RNLE_H_MAX - RNLE_H_SAFE), 0.0), 1.0)
        score = 0.75 * score + 0.25 * over
    if a.get("mean_conf", 1.0) < 0.45:                     # shaky keypoints
        score *= 0.65
    return {"score": round(float(min(score, 1.0)), 3),
            "parts": {"trunk": round(trunk, 3), "reach": round(reach, 3),
                      "legs": round(legs, 3), "low_hands": round(low, 3)}}


class Trigger:
    """Fire once when score >= thr; re-arm after hysteresis + optional cooldown."""
    def __init__(self, threshold=0.65, hyst=0.10, cooldown_s=6.0):
        self.thr, self.hyst, self.cool = threshold, hyst, cooldown_s
        self._armed, self._last = True, 0.0
    def update(self, score: float) -> bool:
        now = time.time()
        if self._armed and score >= self.thr and now - self._last > self.cool:
            self._armed, self._last = False, now
            return True
        if score < self.thr - self.hyst:
            self._armed = True
        return False
