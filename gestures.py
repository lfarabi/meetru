"""Geometric gesture classification with hysteresis and debounce."""
import numpy as np

from tracking import FINGERS, FINGERTIPS, INDEX_MCP, PINKY_MCP, WRIST, palm_center, palm_size

NO_HAND, OPEN_PALM, FIST = "NO_HAND", "OPEN_PALM", "FIST"
ONE, TWO, THREE, FOUR = "ONE", "TWO", "THREE", "FOUR"
FINGER_GESTURES = (ONE, TWO, THREE, FOUR)  # counted raised fingers, thumb folded
THUMB_TIP = 4


def openness(lm: np.ndarray) -> float:
    """Mean fingertip distance from palm center, in palm sizes (scale-invariant)."""
    size = palm_size(lm)
    if size < 1e-6:
        return 0.0
    return float(np.linalg.norm(lm[FINGERTIPS] - palm_center(lm), axis=1).mean() / size)


def raised_fingers(world: np.ndarray, straight_min: float) -> int:
    """How many of index/middle/ring/pinky are raised, from 3D landmarks (unaffected by hand rotation).

    Raised = nearly straight (knuckle→tip distance vs. summed bone lengths) and the tip reaches
    past the middle joint, measured from the wrist.
    """
    count = 0
    for mcp, pip, dip, tip in FINGERS[1:]:
        bones = sum(np.linalg.norm(world[b] - world[a]) for a, b in ((mcp, pip), (pip, dip), (dip, tip)))
        straight = np.linalg.norm(world[tip] - world[mcp]) / max(bones, 1e-9)
        reaches = np.linalg.norm(world[tip] - world[WRIST]) > np.linalg.norm(world[pip] - world[WRIST])
        count += bool(straight > straight_min and reaches)
    return count


def thumb_out(world: np.ndarray) -> float:
    """Thumb tip's sideways position past the index knuckle, in palm lengths.

    > 0: sticks out beside the hand (open palm); < 0: folded over the palm (finger-count gestures).
    """
    across = world[INDEX_MCP] - world[PINKY_MCP]
    across /= max(float(np.linalg.norm(across)), 1e-9)
    return float((world[THUMB_TIP] - world[INDEX_MCP]) @ across / max(palm_size(world), 1e-9))


class GestureClassifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.gesture = NO_HAND
        self._raw = NO_HAND      # hysteresis-filtered per-frame guess
        self._pending = NO_HAND
        self._count = 0
        self.openness = 0.0
        self.fingers = 0
        self.thumb = 0.0

    def update(self, lm: np.ndarray | None, world: np.ndarray | None) -> str:
        if lm is None:
            raw = NO_HAND
        else:
            self.openness = openness(lm)
            self.fingers = raised_fingers(world, self.cfg["finger_straight"])
            self.thumb = thumb_out(world)
            # raised fingers win over openness: one raised finger alone barely lifts the mean openness
            if self.fingers >= 3 and self.thumb > self.cfg["thumb_out"]:
                raw = OPEN_PALM
            elif self.fingers >= 1:
                raw = FINGER_GESTURES[self.fingers - 1]
            elif self.openness < self.cfg["fist_below"]:
                raw = FIST
            else:  # half-closed, nothing raised: keep the previous hand gesture
                raw = self._raw if self._raw != NO_HAND else FIST
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

    Open palm summons the orb. Holding it with the palm turned away from the camera (up, sideways)
    charges it; turning the palm to face the camera throws it: a blast if it was held for at least
    `charge.min_time`, otherwise it just stays. `facing` is how squarely the palm faces the camera
    (1 = straight at it, negative = turned away); two thresholds with a gap between keep the state
    from flickering while the hand rotates. A fist extinguishes the orb.
    """

    def __init__(self, cfg: dict):
        self.o, self.c, self.b = cfg["orb"], cfg["charge"], cfg["blast"]
        self.state = HIDDEN
        self.silent_fade = False  # faded because a finger-count gesture took over (no power-down SFX)
        self.presence = 0.0      # 0..1 fade in/out
        self.charge = 0.0        # 0..1 charge level
        self.blast_power = 0.0   # charge level when thrown
        self.blast_progress = 0.0
        self._t = None
        self._charge_start = self._blast_start = 0.0
        self._cooldown_until = float("-inf")

    def update(self, gesture: str, facing: float | None, t: float) -> str:
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        s = self.state
        aimed = facing is not None and facing > self.b["aim"]
        holding = facing is not None and facing < self.b["hold_below"]

        if s == HIDDEN:
            if gesture == OPEN_PALM and t >= self._cooldown_until:
                self.state = ACTIVE
        elif s == FADING:
            if gesture == OPEN_PALM:
                self.state = ACTIVE
            elif self.presence <= 0:
                self.state = HIDDEN
        elif s in (ACTIVE, CHARGING):
            # while charging, FOUR is still the open hand: the thumb often reads as folded mid-turn
            open_hand = gesture == OPEN_PALM or (s == CHARGING and gesture == FOUR)
            if not open_hand:
                self.state, self.silent_fade = FADING, gesture in FINGER_GESTURES
            elif s == ACTIVE and holding:
                self.state, self._charge_start = CHARGING, t
            elif s == CHARGING:
                held = t - self._charge_start
                self.charge = min(held / self.c["max_time"], 1.0)
                if aimed:
                    if held >= self.c["min_time"]:
                        self.state, self._blast_start, self.blast_power = BLAST, t, self.charge
                    else:
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

    def busy(self, t: float) -> bool:
        """Charging, blasting or in the post-blast cooldown: finger counts seen now are the hand
        mid-turn, not deliberate finger gestures."""
        return self.state in (CHARGING, BLAST) or t < self._cooldown_until
