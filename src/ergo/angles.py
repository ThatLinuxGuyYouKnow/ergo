"""Joint angles + RNLE-aligned H/V proxies from COCO-17 keypoints.

Angles are view-agnostic-ish 2D projections (frontal/sagittal mix), which is
fine for a threshold gate; the VLM gets the frames precisely because 2D
geometry alone is ambiguous (oblique views, occlusion).

RNLE proxies (negotiate with scale): if subject_height_cm is known we convert
pixels -> cm using nose-to-ankle as the body span, then estimate
  H = horizontal hand distance from mid-ankles (cm)
  V = hand height above floor (cm, floor ~ ankle line)
  D = vertical hand travel between two frames (cm)
These line up with SAFELIFT's annotation fields so we can score against them.
"""
from __future__ import annotations
import numpy as np


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    n = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    return float(np.degrees(np.arccos(np.clip(ba @ bc / n, -1, 1))))


def _vis(kps, idxs, thr=0.3) -> float:
    v = kps[idxs, 2]
    return float(np.clip((v.mean() - thr) / (1 - thr), 0, 1)) if v.size else 0.0


def joint_angles(kps: np.ndarray, w: int, h: int) -> dict:
    P = lambda i: np.array([kps[i][0], kps[i][1]], dtype=np.float32)
    l_sh, r_sh, l_hip, r_hip = P(5), P(6), P(11), P(12)
    l_knee, r_knee, l_ank, r_ank = P(13), P(14), P(15), P(16)
    l_wr, r_wr = P(9), P(10)
    sh, hip = (l_sh + r_sh) / 2, (l_hip + r_hip) / 2
    knee, ank = (l_knee + r_knee) / 2, (l_ank + r_ank) / 2
    hands = (l_wr + r_wr) / 2

    vertical = hip + np.array([0.0, -100.0])
    trunk_flex = _angle(vertical, hip, sh)          # 0 upright, ~90 horizontal
    knee_flex = 180.0 - _angle(hip, knee, ank)      # 0 straight leg
    shoulder_elev = _angle(hip, sh, hands)

    # body scale for normalized reach/hand-height terms
    span = float(np.linalg.norm(ank - sh)) + 1e-6
    H = float(abs(hands[0] - ank[0]) / span)        # reach, torso-lengths
    V = float((ank[1] - hands[1]) / span)           # hand height above ankle line

    vis = float(np.mean([_vis(kps, [5, 6, 11, 12]),
                         _vis(kps, [13, 14, 15, 16]),
                         _vis(kps, [9, 10])]))
    return {"trunk_flexion_deg": trunk_flex, "knee_flexion_deg": knee_flex,
            "shoulder_elev_deg": shoulder_elev, "H_norm": H, "V_norm": V,
            "mean_conf": _vis(kps, [0, 5, 6, 11, 12, 13, 14]),
            "visibility": vis}


def rnle_proxies(kps: np.ndarray, frame_h: int, subject_height_cm: float | None) -> dict:
    """Pixel -> cm using the subject's known height. Returns H/V in cm."""
    if not subject_height_cm:
        return {"cm_per_px": None, "H_cm": None, "V_cm": None}
    nose_y = float(kps[0][1]) if kps[0][2] > 0.3 else float(np.nanmin(kps[[5, 6], 1]))
    ank_y = float(np.nanmax(kps[[15, 16], 1]))
    body_px = max(ank_y - nose_y, 1.0) * 1.08      # nose->ankle ~ 93% of stature
    cm_per_px = subject_height_cm / body_px
    hands = (np.array(kps[9][:2], float) + np.array(kps[10][:2], float)) / 2
    ank_x = float((kps[15][0] + kps[16][0]) / 2)
    return {"cm_per_px": round(float(cm_per_px), 4),
            "H_cm": round(abs(float(hands[0]) - ank_x) * cm_per_px, 1),
            "V_cm": round((ank_y - float(hands[1])) * cm_per_px, 1)}
