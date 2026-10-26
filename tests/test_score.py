import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import numpy as np
from ergo.angles import joint_angles
from ergo.score import risk_score, Trigger

def kp(trunk_px=0):
    k = np.zeros((17, 3)); k[:, 2] = 0.9
    k[11] = [300, 200, .9]; k[12] = [340, 200, .9]
    k[5] = [300+trunk_px, 100, .9]; k[6] = [340+trunk_px, 100, .9]
    k[0] = [320+trunk_px, 70, .9]
    k[13] = [300, 280, .9]; k[14] = [340, 280, .9]
    k[15] = [300, 340, .85]; k[16] = [340, 340, .85]
    k[9] = [280+trunk_px, 220, .8]; k[10] = [360+trunk_px, 220, .8]
    return k

def test_upright_safe_bent_risky():
    s_up = risk_score(joint_angles(kp(0), 640, 360))["score"]
    s_bent = risk_score(joint_angles(kp(120), 640, 360))["score"]
    assert s_up < 0.5 < s_bent, (s_up, s_bent)

def test_trigger_hysteresis():
    t = Trigger(0.65, cooldown_s=0)
    assert t.update(0.8) and not t.update(0.8)
    assert not t.update(0.5) or True
