"""Vanishing jutsu: the crossed two-finger seal (seal) plays a smoke puff and, under its
densest moment, swaps you for a captured empty-room plate (or brings you back)."""
import logging
from pathlib import Path

import cv2
import numpy as np

from gestures import raised_fingers
from tracking import INDEX_MCP, MIDDLE_MCP, Hand

log = logging.getLogger(__name__)
INDEX_TIP, MIDDLE_TIP = 8, 12
CAPTURE_FRAMES = 15  # frames median-stacked into the background plate (removes sensor noise)


def _point_seg_dist(p, a, b) -> float:
    ab = b - a
    s = np.clip((p - a) @ ab / max(float(ab @ ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + s * ab)))


def _seg_dist(a0, a1, b0, b1) -> float:
    """Shortest distance between two 2D segments (0 when they cross)."""
    def side(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    if side(a0, a1, b0) != side(a0, a1, b1) and side(b0, b1, a0) != side(b0, b1, a1):
        return 0.0
    return min(_point_seg_dist(a0, b0, b1), _point_seg_dist(a1, b0, b1),
               _point_seg_dist(b0, a0, a1), _point_seg_dist(b1, a0, a1))


def seal_point(hands: list[Hand], straight_min: float) -> np.ndarray | None:
    """Where the seal is made, or None: two hands, each with exactly two fingers raised, the two
    finger pairs crossed at a steep angle (more than 60°) and actually touching."""
    if len(hands) != 2:
        return None
    axes = []
    for h in hands:
        if raised_fingers(h.world, straight_min) != 2:
            return None
        base = (h.lm[INDEX_MCP, :2] + h.lm[MIDDLE_MCP, :2]) / 2
        tip = (h.lm[INDEX_TIP, :2] + h.lm[MIDDLE_TIP, :2]) / 2
        axes.append((base, tip))
    (a0, a1), (b0, b1) = axes
    da, db = a1 - a0, b1 - b0
    la, lb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    if la < 1 or lb < 1 or abs(float(da @ db)) / (la * lb) > 0.5:
        return None
    if _seg_dist(a0, a1, b0, b1) > 0.35 * max(la, lb):
        return None
    return (a0 + a1 + b0 + b1) / 4


def load_greenscreen_clip(path: Path, scale: float = 0.5):
    """Green-screen clip → (frames, fps, peak, anchor). frames = [(bgr uint8, alpha float32 0..1)]
    with the green keyed out and despilled, empty lead-in/tail and non-green frames dropped,
    cropped to where the effect ever appears. peak = index of the densest frame; anchor = (x, y)
    center of its densest part, the point that should land on you."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.int16)
        b, g, r = f[..., 0], f[..., 1], f[..., 2]
        rb = np.maximum(r, b)
        spill = g - rb  # ≈180 on the green screen, ≈0 on white smoke
        if (spill > 100).mean() < 0.3:
            continue  # not a green-screen frame (e.g. the black one at the end)
        alpha = 1 - np.clip((spill - 40) / 100, 0, 1).astype(np.float32)
        despill = np.stack([b, np.minimum(g, rb), r], axis=-1).astype(np.uint8)
        frames.append((despill, alpha))
    cap.release()
    mass = np.array([a.mean() for _, a in frames]) if frames else np.zeros(0)
    keep = np.nonzero(mass > 0.003)[0]
    if len(keep) < 4:
        raise ValueError(f"no green-screen effect found in {path}")
    frames = frames[keep[0]:keep[-1] + 1]
    union = np.max([a for _, a in frames], axis=0) > 0.05
    ys, xs = np.nonzero(union)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    frames = [(c[y0:y1, x0:x1].copy(), a[y0:y1, x0:x1].copy()) for c, a in frames]
    peak = int(np.argmax(mass[keep[0]:keep[-1] + 1]))
    dense = frames[peak][1] > 0.8
    ys, xs = np.nonzero(dense) if dense.any() else np.nonzero(frames[peak][1])
    return frames, fps, peak, (float(xs.mean()), float(ys.mean()))


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Vanish:
    """mix: 0 = you as the camera sees you, 1 = the empty-room plate."""

    def __init__(self, cfg: dict, frame_shape: tuple):
        self.cfg, self.shape = cfg, frame_shape
        self.mix = 0.0
        self._message, self._message_until = "", 0.0
        self.plate = None
        plate_path = Path(cfg["background"])
        if plate_path.exists():
            self.plate = cv2.resize(cv2.imread(str(plate_path)), (frame_shape[1], frame_shape[0]))
        self.clip = None
        try:
            self.clip = load_greenscreen_clip(Path(cfg["smoke"]))
            log.info("Smoke clip %s: %d frames", cfg["smoke"], len(self.clip[0]))
        except Exception as e:
            log.warning("Smoke clip unavailable (%s); vanishing without smoke", e)
        self._since, self._armed = None, True
        self._start, self._at, self._from, self._to = None, None, 0.0, 0.0
        self._capture_at, self._captured = None, []

    @property
    def busy(self) -> bool:
        return self._start is not None

    def message(self, t: float) -> str:
        """Status line for the preview, shown for a few seconds."""
        return self._message if t < self._message_until else ""

    def _say(self, text: str, t: float, seconds: float = 3.0):
        self._message, self._message_until = text, t + seconds

    def start_capture(self, t: float):
        self._capture_at, self._captured = t + self.cfg["capture_delay"], []

    def capture_tick(self, frame: np.ndarray, t: float):
        """Feed the raw camera frame; records the empty room once the countdown ends."""
        if self._capture_at is None:
            return
        if t < self._capture_at:
            self._say(f"Leave the frame: capturing the empty room in {self._capture_at - t:.0f}s", t, 0.2)
            return
        self._captured.append(frame.copy())
        if len(self._captured) >= CAPTURE_FRAMES:
            self.plate = np.median(np.stack(self._captured), axis=0).astype(np.uint8)
            cv2.imwrite(self.cfg["background"], self.plate)
            self._capture_at, self._captured = None, []
            self._say("Empty room captured", t)
            log.info("Background plate saved to %s", self.cfg["background"])

    def update(self, seal: np.ndarray | None, t: float) -> bool:
        """Call once per frame with seal_point(); holding the seal toggles you out/in once.
        Returns True on the frame the seal fires (for its sound)."""
        fired = False
        if seal is None:
            self._since, self._armed = None, True
        else:
            self._since = t if self._since is None else self._since
            if self._armed and not self.busy and t - self._since >= self.cfg["seal_hold"]:
                self._armed, fired = False, True  # let go of the seal before it can fire again
                self._start, self._at, self._from = t, seal, self.mix
                if self.plate is None:
                    self._to = 0.0
                    self._say("No empty-room background yet: press B first", t)
                else:
                    self._to = 1.0 - round(self.mix)
        if self.busy:
            if self.clip:
                frames, fps, peak, _ = self.clip
                i = (t - self._start) * fps
                w, done = _smoothstep((i - peak + 6) / 12), i >= len(frames)  # swap while the puff is densest
            else:
                w = _smoothstep((t - self._start) / 0.5)
                done = w >= 1
            self.mix = self._from + (self._to - self._from) * w
            if done:
                self.mix, self._start = self._to, None
        return fired

    def apply_plate(self, frame: np.ndarray):
        if self.plate is not None and self.mix > 0:
            frame[:] = self.plate if self.mix >= 1 else cv2.addWeighted(frame, 1 - self.mix, self.plate, self.mix, 0)

    def draw_smoke(self, frame: np.ndarray, t: float):
        if not (self.busy and self.clip):
            return
        frames, fps, _, (ax, ay) = self.clip
        i = int((t - self._start) * fps)
        if i >= len(frames):
            return
        color, alpha = frames[i]
        scale = self.cfg["smoke_height"] * frame.shape[0] / color.shape[0]
        w, h = max(1, int(color.shape[1] * scale)), max(1, int(color.shape[0] * scale))
        x0, y0 = int(self._at[0] - ax * scale), int(self._at[1] - ay * scale)  # densest puff on the seal
        fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
        if fx0 >= fx1 or fy0 >= fy1:
            return
        color = cv2.resize(color, (w, h))[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        alpha = cv2.resize(alpha, (w, h))[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        roi = frame[fy0:fy1, fx0:fx1]
        roi[:] = cv2.blendLinear(roi, color, 1 - alpha, alpha)
