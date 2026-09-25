"""Procedural Energy Orb and blast, drawn with NumPy/OpenCV onto the output frame."""
import math

import cv2
import numpy as np

from gestures import BLAST, HIDDEN, OrbMachine
from tracking import MIDDLE_MCP, WRIST, palm_center, palm_size

BLAST_DOWNSCALE = 4  # blast glow is soft, so render it at 1/4 resolution and upscale
ORB_EXTENT = 2.3     # orb render area half-size, in orb radii


def orb_anchor(lm: np.ndarray, cfg: dict) -> tuple[tuple[float, float], float]:
    """Orb center (px), offset from the palm toward the fingers, and its base radius (px)."""
    size = palm_size(lm)
    direction = lm[MIDDLE_MCP, :2] - lm[WRIST, :2]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    center = palm_center(lm)[:2] + direction * cfg["offset"] * size
    return (float(center[0]), float(center[1])), cfg["radius"] * size


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
        rng = np.random.default_rng(seed)
        n = self.o["particles"]
        self.p_angle = rng.uniform(0, 2 * math.pi, n)
        self.p_radius = rng.uniform(1.1, 2.0, n)     # in orb radii
        self.p_speed = rng.uniform(1.0, 3.0, n)      # rad/s, all swirl the same way
        self.p_size = rng.uniform(1.0, 2.5, n)
        self.p_phase = rng.uniform(0, 2 * math.pi, n)
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
        r = radius * scale * (0.6 + 0.4 * m.presence) * (1 + 0.04 * math.sin(2 * math.pi * 2.5 * t))
        self._last_center, self._last_r = center, r
        box = _roi(frame, *center, r * ORB_EXTENT)
        if box is None:
            return
        x0, x1, y0, y1 = box
        xx = (np.arange(x0, x1, dtype=np.float32) - center[0])[None, :] / r
        yy = (np.arange(y0, y1, dtype=np.float32) - center[1])[:, None] / r
        d = np.sqrt(xx * xx + yy * yy)
        theta = np.arctan2(yy, xx)

        glow = self.o["glow"] * (1 + 0.8 * c)
        spin = t * (3 + 9 * c)
        wobble = 0.07 * np.sin(5 * theta + spin) + 0.04 * np.sin(9 * theta - 1.7 * spin)
        ring = np.exp(-((d - 1.0 - wobble) / 0.09) ** 2) * (0.55 + 0.45 * np.sin(3 * theta - 2 * spin))
        window = np.clip(1 - d / ORB_EXTENT, 0, 1) ** 2  # fades the halo to 0 before the render edge
        colored = (np.exp(-(d / 0.9) ** 2) * 0.7
                   + np.exp(-(d / 1.6) ** 2) * 0.4 * glow * window
                   + ring * 0.9
                   + self._orbit_particles(box, center, r, t, c))
        core = np.exp(-(d / 0.42) ** 4) * (0.9 + 0.2 * c)
        tint = self.color * (1 - 0.3 * c) + 0.3 * c  # heats toward white while charging
        layer = (colored[..., None] * tint + core[..., None]) * m.presence
        _screen(frame[y0:y1, x0:x1], layer)

    def _orbit_particles(self, box, center, r, t, c):
        x0, x1, y0, y1 = box
        canvas = np.zeros((y1 - y0, x1 - x0), np.float32)
        ang = self.p_angle + self.p_speed * t * (1 + 2.5 * c)
        rad = r * self.p_radius * (1 - 0.35 * c) * (1 + 0.08 * np.sin(self.p_phase + 3 * t))  # pulled in while charging
        px, py = center[0] - x0 + rad * np.cos(ang), center[1] - y0 + rad * np.sin(ang)
        bright = 0.5 + 0.5 * c
        for x, y, s in zip(px, py, np.maximum(1, self.p_size * r / 40)):
            cv2.circle(canvas, (int(x), int(y)), int(s), bright, -1, cv2.LINE_AA)
        return cv2.GaussianBlur(canvas, (0, 0), 1.2)

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
            _screen(frame[y0:y1, x0:x1], layer[..., None] * (self.color * 0.6 + 0.4))

        flash = min(0.6, 0.35 * strength * (1 - p) ** 6)  # brief whole-frame flash
        if flash > 0.01:
            cv2.convertScaleAbs(frame, dst=frame, alpha=1 - flash, beta=255 * flash)
