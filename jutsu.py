"""Two-hand poses that play green-screen clips.

- Vanishing jutsu: the crossed two-finger seal (seal) plays a smoke puff and, under its
  densest moment, swaps you for a captured empty-room plate (or brings you back).
- Wow: both open hands together under the chin plays a full-frame reaction clip.
"""
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from gestures import raised_fingers
from tracking import INDEX_MCP, MIDDLE_MCP, WRIST, Hand, palm_size

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


def wow_pose(hands: list[Hand], straight_min: float) -> bool:
    """Both hands open (3+ fingers raised), wrists together, fingers pointing up: the hands framing
    the face from under the chin. There's no face tracking, so wrists-together-fingers-up stands
    in for "under the chin"."""
    if len(hands) != 2 or any(raised_fingers(h.world, straight_min) < 3 for h in hands):
        return False
    a, b = hands
    size = (palm_size(a.lm) + palm_size(b.lm)) / 2
    if np.linalg.norm(a.lm[WRIST, :2] - b.lm[WRIST, :2]) > 1.2 * size:
        return False
    for h in hands:  # wrist → middle knuckle within 70° of straight up (image y grows downward);
        d = h.lm[MIDDLE_MCP, :2] - h.lm[WRIST, :2]  # the hands splay out ~57° in the reference photo
        if -d[1] < 0.34 * np.linalg.norm(d):
            return False
    return True


class Clip(NamedTuple):
    frames: list        # [(bgr uint8, alpha uint8)] cropped to where the effect ever appears
    fps: float
    peak: int           # index of the densest frame
    anchor: tuple       # (x, y) center of the densest part of the peak frame, in crop px
    origin: tuple       # (x, y) of the crop within the (scaled) source frame
    source_width: int   # width of the (scaled) source frame, for full-frame placement


def load_greenscreen_clip(path: Path, scale: float = 0.5) -> Clip:
    """Green-screen clip keyed to alpha and despilled, with the empty lead-in/tail and non-green
    frames (e.g. a black last frame) dropped, cropped to where the effect ever appears.

    The key adapts to the clip: how green the screen is (green minus the larger of red/blue) is
    measured on the frame borders, so a dull green screen keys as cleanly as a vivid one."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        raw.append(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    cap.release()
    if not raw:
        raise ValueError(f"cannot read {path}")

    def spill_of(f):
        f = f.astype(np.int16)
        return f[..., 1] - np.maximum(f[..., 0], f[..., 2])

    key = np.percentile([np.median(np.concatenate([s[:4].ravel(), s[-4:].ravel(), s[:, :4].ravel(), s[:, -4:].ravel()]))
                         for s in map(spill_of, raw[::5])], 90)
    if key < 40:
        raise ValueError(f"{path} has no green screen")

    frames, source_width = [], raw[0].shape[1]
    for f in raw:
        spill = spill_of(f)
        if (spill > 0.6 * key).mean() < 0.3:
            continue  # not a green-screen frame
        alpha = 255 - np.clip((spill - 0.3 * key) / (0.35 * key) * 255, 0, 255)
        despill = f.copy()
        despill[..., 1] = np.minimum(f[..., 1], np.maximum(f[..., 0], f[..., 2]))
        frames.append((despill, alpha.astype(np.uint8)))
    del raw
    mass = np.array([a.mean() / 255 for _, a in frames]) if frames else np.zeros(0)
    keep = np.nonzero(mass > 0.003)[0]
    if len(keep) < 4:
        raise ValueError(f"no green-screen effect found in {path}")
    frames, mass = frames[keep[0]:keep[-1] + 1], mass[keep[0]:keep[-1] + 1]
    union = np.max([a for _, a in frames], axis=0) > 12
    ys, xs = np.nonzero(union)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    frames = [(c[y0:y1, x0:x1].copy(), a[y0:y1, x0:x1].copy()) for c, a in frames]
    peak = int(np.argmax(mass))
    dense = frames[peak][1] > 204
    ys, xs = np.nonzero(dense) if dense.any() else np.nonzero(frames[peak][1])
    return Clip(frames, fps, peak, (float(xs.mean()), float(ys.mean())), (int(x0), int(y0)), source_width)


def draw_clip(frame: np.ndarray, clip: Clip, i: int, x0: int, y0: int, scale: float):
    """Alpha-blend clip frame i onto `frame`, its crop's top-left at (x0, y0), resized by `scale`."""
    color, alpha = clip.frames[i]
    w, h = max(1, int(color.shape[1] * scale)), max(1, int(color.shape[0] * scale))
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if fx0 >= fx1 or fy0 >= fy1:
        return
    color = cv2.resize(color, (w, h))[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    alpha = cv2.resize(alpha, (w, h))[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32) * (1 / 255)
    roi = frame[fy0:fy1, fx0:fx1]
    roi[:] = cv2.blendLinear(roi, color, 1 - alpha, alpha)


def _load(path: str, what: str) -> Clip | None:
    try:
        clip = load_greenscreen_clip(Path(path))
        log.info("%s clip %s: %d frames", what, path, len(clip.frames))
        return clip
    except Exception as e:  # media isn't in git, so a fresh clone lands here
        log.warning("%s clip unavailable (%s)", what, e)
        return None


class HoldTrigger:
    """Fires once when a pose has been held `hold` seconds; re-arms only after the pose is let go."""

    def __init__(self, hold: float):
        self.hold = hold
        self._since, self._armed = None, True

    def update(self, active: bool, t: float, allowed: bool = True) -> bool:
        if not active:
            self._since, self._armed = None, True
            return False
        self._since = t if self._since is None else self._since
        if self._armed and allowed and t - self._since >= self.hold:
            self._armed = False
            return True
        return False


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
        self.clip = _load(cfg["smoke"], "Smoke")
        self._trigger = HoldTrigger(cfg["seal_hold"])
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
        fired = self._trigger.update(seal is not None, t, allowed=not self.busy)
        if fired:
            self._start, self._at, self._from = t, seal, self.mix
            if self.plate is None:
                self._to = 0.0
                self._say("No empty-room background yet: press B first", t)
            else:
                self._to = 1.0 - round(self.mix)
        if self.busy:
            if self.clip:
                i = (t - self._start) * self.clip.fps
                w, done = _smoothstep((i - self.clip.peak + 6) / 12), i >= len(self.clip.frames)  # swap under the densest puff
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
        i = int((t - self._start) * self.clip.fps)
        if i >= len(self.clip.frames):
            return
        scale = self.cfg["smoke_height"] * frame.shape[0] / self.clip.frames[0][0].shape[0]
        ax, ay = self.clip.anchor
        draw_clip(frame, self.clip, i, int(self._at[0] - ax * scale), int(self._at[1] - ay * scale), scale)  # densest puff on the seal


class PoseClip:
    """Plays a green-screen clip full-frame, as it was framed, once each time a pose is held."""

    def __init__(self, cfg: dict, what: str):
        self.clip = _load(cfg["clip"], what)
        self._trigger = HoldTrigger(cfg["hold"])
        self._start = None

    @property
    def busy(self) -> bool:
        return self._start is not None

    def update(self, posed: bool, t: float, allowed: bool = True) -> bool:
        """Returns True on the frame the pose fires (for its sound)."""
        fired = self._trigger.update(posed, t, allowed and not self.busy)
        if fired and self.clip:
            self._start = t
        if self.busy and (t - self._start) * self.clip.fps >= len(self.clip.frames):
            self._start = None
        return fired

    def draw(self, frame: np.ndarray, t: float):
        if not self.busy:
            return
        i = int((t - self._start) * self.clip.fps)
        if i >= len(self.clip.frames):
            return
        scale = frame.shape[1] / self.clip.source_width
        ox, oy = self.clip.origin
        draw_clip(frame, self.clip, i, int(ox * scale), int(oy * scale), scale)
