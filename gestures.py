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


HIDDEN, ACTIVE, CHARGING, BLAST, FADING = "HIDDEN", "ACTIVE", "CHARGING", "BLAST", "FADING"


class OrbMachine:
    """Explicit orb state machine.

    Driven by the debounced gesture, plus raw openness so a fast fist→open release can fire
    the blast without waiting for debounce.
    """

    def __init__(self, cfg: dict):
        self.g, self.o, self.c, self.b = cfg["gesture"], cfg["orb"], cfg["charge"], cfg["blast"]
        self.state = HIDDEN
        self.presence = 0.0      # 0..1 fade in/out
        self.charge = 0.0        # 0..1 charge level
        self.blast_power = 0.0   # charge level at release
        self.blast_progress = 0.0
        self._t = None
        self._charge_start = self._blast_start = 0.0
        self._cooldown_until = float("-inf")
        self._left_fist_at = None
        self._open_frames = 0

    def update(self, gesture: str, openness: float | None, t: float) -> str:
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        s = self.state

        if s == HIDDEN:
            if gesture == OPEN_PALM and t >= self._cooldown_until:
                self.state = ACTIVE
        elif s == FADING:
            if gesture == OPEN_PALM:
                self.state = ACTIVE
            elif self.presence <= 0:
                self.state = HIDDEN
        elif s == ACTIVE:
            if gesture == NO_HAND:
                self.state = FADING
            elif gesture == FIST:
                self.state, self._charge_start = CHARGING, t
                self._left_fist_at, self._open_frames = None, 0
        elif s == CHARGING:
            held = t - self._charge_start
            self.charge = min(held / self.c["max_time"], 1.0)
            if gesture == NO_HAND:
                self.state = FADING
            elif self._released(openness, t) and held >= self.c["min_time"]:
                self.state, self._blast_start, self.blast_power = BLAST, t, self.charge
            elif gesture == OPEN_PALM:
                self.state = ACTIVE
        elif s == BLAST:
            self.blast_progress = (t - self._blast_start) / self.b["duration"]
            if self.blast_progress >= 1:
                self.state, self.presence = HIDDEN, 0.0
                self._cooldown_until = self._blast_start + self.b["cooldown"]

        if self.state in (ACTIVE, CHARGING):
            self.presence = min(1.0, self.presence + dt / self.o["spawn_time"])
        elif self.state == FADING:
            self.presence = max(0.0, self.presence - dt / self.o["fade_time"])
        if self.state not in (CHARGING, BLAST):
            self.charge = 0.0
        if self.state != BLAST:
            self.blast_progress = 0.0
        return self.state

    def _released(self, openness: float | None, t: float) -> bool:
        """True once the hand went from fist to open within the release window."""
        if openness is None or openness < self.g["fist_below"]:
            self._left_fist_at, self._open_frames = None, 0
            return False
        if self._left_fist_at is None:
            self._left_fist_at = t
        self._open_frames = self._open_frames + 1 if openness > self.g["open_above"] else 0
        return (self._open_frames >= self.b["confirm_frames"]
                and t - self._left_fist_at <= self.b["release_window"])
