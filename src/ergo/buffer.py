"""Rolling frame buffer -> event clip (pre-roll + post-roll), written by OpenCV 5."""
from __future__ import annotations
import collections, os, time
import cv2


class ClipBuffer:
    def __init__(self, pre_s=4.0, post_s=3.0, fps=15, out_dir="data/clips"):
        self.pre = collections.deque(maxlen=int(pre_s * fps))
        self.post_n, self.fps, self.out_dir = int(post_s * fps), fps, out_dir
        self._active, self._frames = False, []

    def push(self, frame):
        """Always-on pre-roll ring buffer (the passive part)."""
        self.pre.append(frame.copy())

    def start(self):
        """Event declared: freeze pre-roll, begin collecting post-roll."""
        if self._active:
            return
        self._active, self._frames = True, [f.copy() for f in self.pre]

    @property
    def active(self) -> bool:
        return self._active

    def feed(self, frame) -> str | None:
        """Append during an event; returns the clip path once post-roll is done."""
        if not self._active:
            return None
        self._frames.append(frame.copy())
        if len(self._frames) >= len(self.pre) + self.post_n:
            self._active = False
            return self._flush()
        return None

    @property
    def frames(self):
        return self._frames

    def _flush(self) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"event_{int(time.time()*1000)}.mp4")
        h, w = self._frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for f in self._frames:
            vw.write(f)
        vw.release()
        return path
