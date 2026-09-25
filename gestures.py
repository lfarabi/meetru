"""Geometric gesture classification with hysteresis and debounce."""
import numpy as np

from tracking import FINGERTIPS, palm_center, palm_size

NO_HAND, OPEN_PALM, FIST = "NO_HAND", "OPEN_PALM", "FIST"


def openness(lm: np.ndarray) -> float:
    """Mean fingertip distance from palm center, in palm sizes (scale-invariant)."""
    size = palm_size(lm)
    if size < 1e-6:
        return 0.0
    return float(np.linalg.norm(lm[FINGERTIPS] - palm_center(lm), axis=1).mean() / size)


class GestureClassifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.gesture = NO_HAND
        self._raw = NO_HAND      # hysteresis-filtered per-frame guess
        self._pending = NO_HAND
        self._count = 0
        self.openness = 0.0

    def update(self, lm: np.ndarray | None) -> str:
        if lm is None:
            raw = NO_HAND
        else:
            self.openness = openness(lm)
            if self.openness < self.cfg["fist_below"]:
                raw = FIST
            elif self.openness > self.cfg["open_above"]:
                raw = OPEN_PALM
            else:  # dead zone: keep previous hand gesture, default to open
                raw = self._raw if self._raw != NO_HAND else OPEN_PALM
        self._raw = raw

        if raw == self.gesture:
            self._count = 0
            return self.gesture
        if raw != self._pending:
            self._pending, self._count = raw, 0
        self._count += 1
        needed = self.cfg["lost_frames"] if raw == NO_HAND else self.cfg["debounce_frames"]
        if self._count >= needed:
            self.gesture, self._count = raw, 0
        return self.gesture
