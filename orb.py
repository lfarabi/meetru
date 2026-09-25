"""Procedural Energy Orb (Orb-style) and blast, drawn with NumPy/OpenCV onto the output frame."""
import math

import cv2
import numpy as np

from gestures import BLAST, HIDDEN, OrbMachine
from tracking import FINGERS, MIDDLE_MCP, PALM_OUTLINE, PALM_POINTS, WRIST, palm_size

BLAST_DOWNSCALE = 4   # blast glow is soft, so render it at 1/4 resolution and upscale
ORB_EXTENT = 2.1      # render area half-size, in orb radii
RENDER_RADIUS = 60    # orb is rendered at most this many px in radius, then upscaled
N_STREAKS = 45        # tangled light arcs inside the sphere
N_SWOOSHES = 5        # tapered crescents orbiting outside the sphere
SWOOSH_SEGMENTS = 10  # a swoosh is drawn as segments thinning from head to tail


def orb_anchor(lm: np.ndarray, normal: np.ndarray, cfg: dict) -> tuple[tuple[float, float], float]:
    """Orb center (px) hovering off the middle of the palm along the palm normal, and its base radius (px).

    Only the normal's x/y shift the image position: a palm facing up lifts the orb above the hand,
    facing down hangs it below. Within `center_lock` of facing the camera (or away) the orb snaps to
    the palm middle, so small tilts and depth noise don't make it slide off the palm.
    """
    wrist = lm[WRIST, :2]
    knuckles = lm[PALM_POINTS[1:], :2].mean(axis=0)
    radius = cfg["radius"] * palm_size(lm)
    lock = np.clip((abs(normal[2]) - cfg["center_lock"]) / (1 - cfg["center_lock"]) * 1.25, 0, 1)
    center = wrist + (knuckles - wrist) * cfg["palm_position"] + normal[:2] * cfg["hover"] * radius * (1 - lock)
    return (float(center[0]), float(center[1])), radius


def occlude_by_hand(frame: np.ndarray, original: np.ndarray, lm: np.ndarray, strength: float):
    """Put the hand back in front of the effect (orb behind the hand), using a silhouette from landmarks."""
    pts = lm[:, :2].astype(np.int32)
    thick = max(2, int(np.linalg.norm(lm[MIDDLE_MCP, :2] - lm[WRIST, :2]) * 0.28))
    x0, y0 = np.maximum(pts.min(axis=0) - thick, 0)
    x1, y1 = np.minimum(pts.max(axis=0) + thick + 1, (frame.shape[1], frame.shape[0]))
    if x0 >= x1 or y0 >= y1:
        return
    local = pts - (x0, y0)
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(local[PALM_OUTLINE]), 1.0, cv2.LINE_AA)
    cv2.polylines(mask, [local[f] for f in FINGERS], False, 1.0, thick, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), thick * 0.25) * strength
    roi = frame[y0:y1, x0:x1]
    roi[:] = cv2.blendLinear(roi, original[y0:y1, x0:x1], 1 - mask, mask)


def _roi(frame: np.ndarray, cx: float, cy: float, half: float):
    h, w = frame.shape[:2]
    x0, x1 = max(int(cx - half), 0), min(int(cx + half) + 1, w)
    y0, y1 = max(int(cy - half), 0), min(int(cy + half) + 1, h)
    return (x0, x1, y0, y1) if x0 < x1 and y0 < y1 else None


def _screen(roi: np.ndarray, layer: np.ndarray):
    """Screen blend (adds light without clipping to flat white) a 0..1 float layer into a uint8 ROI.

    Done in uint8 with cv2 ops: 255 - (255-a)(255-b)/255, several times faster than float NumPy.
    """
    inv_layer = cv2.bitwise_not(cv2.convertScaleAbs(layer, alpha=255))
    roi[:] = cv2.bitwise_not(cv2.multiply(cv2.bitwise_not(roi), inv_layer, scale=1 / 255))


class EnergyOrb:
    def __init__(self, cfg: dict, seed: int = 0):
        self.o, self.charge_cfg = cfg["orb"], cfg["charge"]
        self.color = np.array(self.o["color_bgr"], np.float32) / 255
        self.line_tint = self.color * 0.3 + 0.7  # streaks, sparks and swoosh edges glow near-white
        rng = np.random.default_rng(seed)

        n = self.o["particles"]
        self.p_angle = rng.uniform(0, 2 * math.pi, n)
        self.p_radius = rng.uniform(1.1, 1.9, n)     # in orb radii
        self.p_speed = rng.uniform(1.0, 3.0, n)      # rad/s, all swirl the same way
        self.p_size = rng.uniform(0.8, 2.0, n)
        self.p_phase = rng.uniform(0, 2 * math.pi, n)

        # each streak is an arc of a tilted ellipse (a great circle seen at an angle), spinning
        self.s_axis = rng.uniform(0.45, 0.95, N_STREAKS)
        self.s_flat = rng.uniform(0.1, 0.7, N_STREAKS)
        self.s_angle = rng.uniform(0, 360, N_STREAKS)
        self.s_speed = rng.uniform(40, 140, N_STREAKS) * rng.choice([-1, 1], N_STREAKS)  # deg/s
        self.s_start = rng.uniform(0, 360, N_STREAKS)
        self.s_span = rng.uniform(90, 250, N_STREAKS)
        self.s_bright = rng.uniform(0.35, 1.0, N_STREAKS)
        self.s_offset = rng.uniform(-0.12, 0.12, (N_STREAKS, 2))

        self.w_tilt = np.linspace(0, 360, N_SWOOSHES, endpoint=False) + rng.uniform(-20, 20, N_SWOOSHES)
        self.w_a = rng.uniform(1.4, 1.7, N_SWOOSHES)
        self.w_b = rng.uniform(0.85, 1.15, N_SWOOSHES)
        self.w_phase = rng.uniform(0, 360, N_SWOOSHES)
        self.w_speed = rng.uniform(110, 170, N_SWOOSHES)
        self.w_span = rng.uniform(110, 160, N_SWOOSHES)

        self._prev_state = HIDDEN
        self._last_center, self._last_r = None, None
        self._blast_center, self._blast_r = None, None

    def draw(self, frame: np.ndarray, m: OrbMachine, center, radius: float, t: float):
        if m.state == BLAST:
            if self._prev_state != BLAST:
                self._blast_center = self._last_center or center
                self._blast_r = self._last_r or radius
            self._draw_blast(frame, m)
        elif m.state != HIDDEN:
            self._draw_orb(frame, m, center, radius, t)
        self._prev_state = m.state

    def _draw_orb(self, frame, m, center, radius, t):
        c = m.charge
        scale = 1 + (self.charge_cfg["max_scale"] - 1) * c ** 1.5
        r = radius * scale * (0.6 + 0.4 * m.presence) * (1 + 0.03 * math.sin(2 * math.pi * 2.5 * t))
        self._last_center, self._last_r = center, r
        box = _roi(frame, *center, r * ORB_EXTENT)
        if box is None:
            return
        x0, x1, y0, y1 = box
        full = (x1 - x0, y1 - y0)
        # render at a fixed internal size and upscale: cost stays flat as the orb grows while charging
        ds = max(1.0, r / RENDER_RADIUS)
        shape = (math.ceil(full[1] / ds), math.ceil(full[0] / ds))
        cx, cy, rl = (center[0] - x0) / ds, (center[1] - y0) / ds, r / ds
        xx = (np.arange(shape[1], dtype=np.float32) - cx)[None, :] / rl
        yy = (np.arange(shape[0], dtype=np.float32) - cy)[:, None] / rl
        d = np.sqrt(xx * xx + yy * yy)

        spin = 1 + 2.5 * c
        inside = np.clip((1.05 - d) / 0.1, 0, 1)
        streaks = self._streaks(shape, cx, cy, rl, t, spin, 0.75 + 0.5 * c) * inside
        streaks = cv2.GaussianBlur(streaks, (0, 0), 0.6) + cv2.GaussianBlur(streaks, (0, 0), rl * 0.05) * 0.8
        swoosh = self._swooshes(shape, cx, cy, rl, t, spin, 1 - 0.12 * c)
        swoosh = cv2.GaussianBlur(swoosh, (0, 0), 0.8) + cv2.GaussianBlur(swoosh, (0, 0), rl * 0.08)
        sparks = self._orbit_particles(shape, cx, cy, rl, t, c)

        window = np.clip(1 - d / ORB_EXTENT, 0, 1) ** 2
        halo = np.exp(-(d / 1.3) ** 2) * 0.55 * self.o["glow"] * (1 + 0.8 * c) * window
        core = np.exp(-(d / 0.28) ** 2) * (1.0 + 0.3 * c) + np.exp(-(d / 0.6) ** 2) * 0.35
        light = ((halo + swoosh * 0.9)[..., None] * self.color
                 + (streaks + swoosh * 0.25 + sparks)[..., None] * self.line_tint
                 + core[..., None]) * m.presence

        # sphere, swooshes and halo are painted over the hand (not only added as light) so they
        # stay saturated blue on skin instead of washing out to pale lavender
        alpha = np.clip(np.clip((1 - d) / 0.08, 0, 1) * self.o["opacity"]
                        + np.clip(swoosh, 0, 1) * 0.85 + halo * 0.5, 0, 1) * m.presence
        body = self.color * (0.7 + 0.3 * np.exp(-(d / 0.6) ** 2))[..., None]

        if ds > 1:
            light, alpha, body = (cv2.resize(a, full, interpolation=cv2.INTER_LINEAR) for a in (light, alpha, body))
        roi = frame[y0:y1, x0:x1]
        roi[:] = cv2.blendLinear(roi, cv2.convertScaleAbs(body, alpha=255), 1 - alpha, alpha)
        _screen(roi, light)

    def _streaks(self, shape, cx, cy, r, t, spin, bright):
        canvas = np.zeros(shape, np.float32)
        thick = 1
        for i in range(N_STREAKS):
            a = r * self.s_axis[i]
            rot = self.s_angle[i] + self.s_speed[i] * t * spin
            start = self.s_start[i] + self.s_speed[i] * 0.6 * t * spin
            pos = (int(cx + self.s_offset[i, 0] * r), int(cy + self.s_offset[i, 1] * r))
            cv2.ellipse(canvas, pos, (int(a), max(1, int(a * self.s_flat[i]))), rot, start,
                        start + self.s_span[i], float(self.s_bright[i] * bright), thick, cv2.LINE_AA)
        return canvas

    def _swooshes(self, shape, cx, cy, r, t, spin, shrink):
        canvas = np.zeros(shape, np.float32)
        pos = (int(cx), int(cy))
        for k in range(N_SWOOSHES):
            axes = (int(r * self.w_a[k] * shrink), int(r * self.w_b[k] * shrink))
            head = self.w_phase[k] + self.w_speed[k] * t * spin
            seg = self.w_span[k] / SWOOSH_SEGMENTS
            for j in range(SWOOSH_SEGMENTS):
                f = 1 - j / SWOOSH_SEGMENTS  # 1 at the head, fading toward the tail
                cv2.ellipse(canvas, pos, axes, self.w_tilt[k], head - (j + 1) * seg, head - j * seg,
                            0.25 + 0.75 * f, max(1, int(r * 0.14 * f ** 1.5)), cv2.LINE_AA)
        return canvas

    def _orbit_particles(self, shape, cx, cy, r, t, c):
        canvas = np.zeros(shape, np.float32)
        ang = self.p_angle + self.p_speed * t * (1 + 2.5 * c)
        rad = r * self.p_radius * (1 - 0.3 * c) * (1 + 0.08 * np.sin(self.p_phase + 3 * t))  # pulled in while charging
        px, py = cx + rad * np.cos(ang), cy + rad * np.sin(ang)
        bright = 0.6 + 0.4 * c
        for x, y, s in zip(px, py, np.maximum(1, self.p_size * r / 50)):
            cv2.circle(canvas, (int(x), int(y)), int(s), bright, -1, cv2.LINE_AA)
        return cv2.GaussianBlur(canvas, (0, 0), 1.0)

    def _draw_blast(self, frame, m):
        p, power = m.blast_progress, m.blast_power
        (cx, cy), r0 = self._blast_center, self._blast_r
        ease = 1 - (1 - p) ** 3
        strength = 0.6 + power
        shock_r = r0 * (1 + (2 + 4 * power) * ease)
        box = _roi(frame, cx, cy, shock_r * 1.3)
        if box is not None:
            x0, x1, y0, y1 = box
            k = BLAST_DOWNSCALE
            small = np.zeros(((y1 - y0 + k - 1) // k, (x1 - x0 + k - 1) // k), np.float32)
            sc = (int((cx - x0) / k), int((cy - y0) / k))
            cv2.circle(small, sc, int(shock_r / k), (1 - p) ** 2 * strength,
                       max(1, int(r0 * 0.35 * (1 - p) / k) + 1), cv2.LINE_AA)
            yy, xx = np.ogrid[:small.shape[0], :small.shape[1]]
            flash_r = r0 * (0.8 + 0.8 * ease) / k
            small += np.exp(-((xx - sc[0]) ** 2 + (yy - sc[1]) ** 2) / flash_r ** 2) * (1 - p) ** 4 * 1.5 * strength
            fly = r0 * (1 + ease * (1.5 + 3 * power) * self.p_radius)
            for a, dist, s in zip(self.p_angle, fly, self.p_size):
                pos = (int(sc[0] + dist * math.cos(a) / k), int(sc[1] + dist * math.sin(a) / k))
                cv2.circle(small, pos, max(1, int(s * r0 / 30 / k)), (1 - p) * strength, -1, cv2.LINE_AA)
            small = cv2.GaussianBlur(small, (0, 0), 2)
            layer = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            _screen(frame[y0:y1, x0:x1], layer[..., None] * self.line_tint)

        flash = min(0.6, 0.35 * strength * (1 - p) ** 6)  # brief whole-frame flash
        if flash > 0.01:
            cv2.convertScaleAbs(frame, dst=frame, alpha=1 - flash, beta=255 * flash)
