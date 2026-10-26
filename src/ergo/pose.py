"""OpenCV 5 pose via cv2.dnn ONNX.

Primary backend: YOLOv8-pose ONNX (single-shot, COCO-17 keypoints).
  model: models/yolov8n-pose.onnx  (13.5 MB, output 1x56x8400)
  output layout per anchor: [cx, cy, w, h, conf, (x,y,conf)*17] in letterbox px
Also accepts heatmap-style HPE exports (1,17,H,W) as a fallback decoder,
and a deterministic mock backend so the stack runs with no model at all.

OpenCV 5 is required: OpenCV 4.6 cannot parse the YOLOv8 `Split` node.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass

COCO_NAMES = ["nose","l_eye","r_eye","l_ear","r_ear","l_sh","r_sh","l_elb","r_elb",
              "l_wr","r_wr","l_hip","r_hip","l_knee","r_knee","l_ank","r_ank"]
# skeleton edges for the overlay (COCO-17)
COCO_EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
              (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]

@dataclass
class PoseResult:
    keypoints: np.ndarray  # (17,3) x,y in original-frame pixels + confidence
    backend: str           # "yolov8-pose" | "heatmap" | "mock"
    person_conf: float = 0.0
    bbox: tuple | None = None

def letterbox(img: np.ndarray, size=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size[1] - nw) // 2, (size[0] - nh) // 2
    out = cv2.copyMakeBorder(resized, dh, size[0] - nh - dh, dw, size[1] - nw - dw,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, dw, dh

def draw_pose(frame: np.ndarray, kps: np.ndarray, edges=COCO_EDGES,
              thr: float = 0.35) -> np.ndarray:
    out = frame.copy()
    for a, b in edges:
        if kps[a][2] > thr and kps[b][2] > thr:
            cv2.line(out, (int(kps[a][0]), int(kps[a][1])),
                     (int(kps[b][0]), int(kps[b][1])), (0, 220, 90), 2)
    for i in range(17):
        if kps[i][2] > thr:
            cv2.circle(out, (int(kps[i][0]), int(kps[i][1])), 3, (30, 120, 255), -1)
    return out

class PoseEstimator:
    def __init__(self, model_path: str | None = None, conf_thr: float = 0.35,
                 nms_thr: float = 0.45, input_size=(640, 640)):
        self.conf_thr, self.nms_thr, self.input_size = conf_thr, nms_thr, input_size
        self.net = None
        if model_path:
            self.net = cv2.dnn.readNetFromONNX(model_path)  # OpenCV 5 DNN
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @property
    def backend(self) -> str:
        return "yolov8-pose" if self.net is not None else "mock"

    def infer(self, frame: np.ndarray) -> PoseResult:
        if self.net is None:
            return self._mock_infer(frame)
        return self._dnn_infer(frame)

    def _dnn_infer(self, frame: np.ndarray) -> PoseResult:
        h, w = frame.shape[:2]
        img, r, dw, dh = letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, self.input_size,
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = np.asarray(out, dtype=np.float32)

        if out.ndim == 4:                      # heatmap-style export (1,17,H,W)
            return self._decode_heatmap(out, w, h)

        pred = np.squeeze(out)                 # (56, 8400) or (8400, 56)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        if pred.shape[1] < 5 + 17 * 2:         # not a pose head
            return PoseResult(np.zeros((17, 3), np.float32), "unsupported")

        scores = pred[:, 4]
        keep = scores >= self.conf_thr
        if not keep.any():
            return PoseResult(np.zeros((17, 3), np.float32), "yolov8-pose", 0.0)
        pred, scores = pred[keep], scores[keep]

        xywh = pred[:, :4]
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                          xywh[:, 2], xywh[:, 3]], axis=1)
        idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                self.conf_thr, self.nms_thr)
        idxs = np.array(idxs).flatten() if len(idxs) else np.array([], int)
        if idxs.size == 0:
            return PoseResult(np.zeros((17, 3), np.float32), "yolov8-pose",
                              float(scores.max()))
        # single-subject floor analysis: keep the most confident person
        i = idxs[np.argmax(scores[idxs])]

        kps = pred[i, 5:5 + 17 * 3].reshape(17, 3).copy()
        kps[:, 0] = (kps[:, 0] - dw) / r       # undo letterbox
        kps[:, 1] = (kps[:, 1] - dh) / r
        kps[:, 0] = np.clip(kps[:, 0], 0, w - 1)
        kps[:, 1] = np.clip(kps[:, 1], 0, h - 1)
        b = boxes[i]
        bbox = tuple(float(v) for v in [(b[0] - dw) / r, (b[1] - dh) / r,
                                        b[2] / r, b[3] / r])
        return PoseResult(kps, "yolov8-pose", float(scores[i]), bbox)

    def _decode_heatmap(self, out, w, h) -> PoseResult:
        if out.shape[1] == 17:                 # (1,17,H,W)
            hm = out[0]
        elif out.shape[3] == 17:               # (1,H,W,17)
            hm = out[0].transpose(2, 0, 1)
        else:
            return PoseResult(np.zeros((17, 3), np.float32), "unsupported")
        kps = np.zeros((17, 3), np.float32)
        for i in range(17):
            _, conf, _, loc = cv2.minMaxLoc(hm[i])
            kps[i] = [loc[0] / hm.shape[1] * w, loc[1] / hm.shape[0] * h, float(conf)]
        return PoseResult(kps, "heatmap", float(kps[:, 2].mean()))

    def _mock_infer(self, frame: np.ndarray) -> PoseResult:
        """Deterministic silhouette heuristic so the pipeline runs with no model."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, th = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx, cy, bw, bh = (w / 2, h / 2, w * 0.3, h * 0.6)
        if cnts:
            x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            cx, cy = x + bw / 2, y + bh / 2
        bend = float(np.clip((bw / max(bh, 1) - 0.35) * 2.2, 0, 1))
        kps = np.zeros((17, 3), np.float32)
        lean = bend * bw * 0.45
        kps[0] = [cx + lean, cy - bh * 0.42, 0.9]
        kps[5] = [cx - bw * 0.18 + lean * 0.6, cy - bh * 0.22, 0.9]
        kps[6] = [cx + bw * 0.18 + lean * 0.6, cy - bh * 0.22, 0.9]
        kps[11] = [cx - bw * 0.12, cy + bh * 0.10, 0.9]
        kps[12] = [cx + bw * 0.12, cy + bh * 0.10, 0.9]
        kps[13] = [cx - bw * 0.12, cy + bh * 0.32, 0.85]
        kps[14] = [cx + bw * 0.12, cy + bh * 0.32, 0.85]
        kps[15] = [cx - bw * 0.12, cy + bh * 0.48, 0.8]
        kps[16] = [cx + bw * 0.12, cy + bh * 0.48, 0.8]
        kps[7] = [cx - bw * 0.25 + lean * 0.8, cy - bh * 0.02, 0.8]
        kps[8] = [cx + bw * 0.25 + lean * 0.8, cy - bh * 0.02, 0.8]
        kps[9] = [cx - bw * 0.28 + lean, cy + bh * 0.16, 0.75]
        kps[10] = [cx + bw * 0.28 + lean, cy + bh * 0.16, 0.75]
        return PoseResult(kps, "mock", 0.5)
