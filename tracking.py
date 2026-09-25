"""Webcam hand landmarks via MediaPipe, plus palm geometry and smoothing."""
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

log = logging.getLogger(__name__)

WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 5, 9, 13, 17
FINGERTIPS = [8, 12, 16, 20]
PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
PALM_OUTLINE = [0, 1, 2, 5, 9, 13, 17]
CURL_POINTS = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]  # thumb + finger joints and tips
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Hand(NamedTuple):
    lm: np.ndarray      # (21, 3) pixels, z scaled like x
    world: np.ndarray   # (21, 3) meters around the hand center, camera axes (x right, y down, z away)


def palm_normal(world: np.ndarray, prev_side: float = 1.0, min_curl: float = 0.003) -> tuple[np.ndarray, float]:
    """Unit vector out of the palm (camera axes; z < 0 = palm faces the camera), and the side sign used.

    The wrist/index/pinky plane gives the palm's orientation but not which face is the palm: that
    depends on left vs right hand, and MediaPipe's handedness label flips in foreshortened poses
    (palm up/down). Thumb and fingers always curl toward the palm, so the side they sit on picks
    the palm face. A near-flat hand (curl < min_curl meters) keeps the previous side.
    """
    n = np.cross(world[INDEX_MCP] - world[WRIST], world[PINKY_MCP] - world[WRIST])
    n /= max(float(np.linalg.norm(n)), 1e-9)
    curl = float(np.mean((world[CURL_POINTS] - world[WRIST]) @ n))
    side = float(np.sign(curl)) if abs(curl) >= min_curl else prev_side
    return n * side, side


def palm_center(lm: np.ndarray) -> np.ndarray:
    return lm[PALM_POINTS].mean(axis=0)


def palm_size(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]))


class Smoother:
    """Exponential moving average over the whole landmark array."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, x: np.ndarray | None) -> np.ndarray | None:
        if x is None:
            self.value = None  # hand lost: don't glide from a stale position when it returns
        elif self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class HandTracker:
    def __init__(self, cfg: dict):
        model = Path(cfg["model_path"])
        if not model.exists():
            raise FileNotFoundError(f"MediaPipe model not found at {model}; see README.md for the download command")
        self._landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg["max_hands"],
            min_hand_detection_confidence=cfg["detection_confidence"],
            min_hand_presence_confidence=cfg["detection_confidence"],
            min_tracking_confidence=cfg["tracking_confidence"],
        ))
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, t_ms: int) -> list[Hand]:
        t_ms = max(t_ms, self._last_ts + 1)  # MediaPipe requires strictly increasing timestamps
        self._last_ts = t_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
        h, w = frame_bgr.shape[:2]
        return [Hand(np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float32),
                     np.array([(p.x, p.y, p.z) for p in wl], dtype=np.float32))
                for lm, wl in zip(result.hand_landmarks, result.hand_world_landmarks)]

    def close(self):
        self._landmarker.close()


def pick_primary(hands: list[Hand], prev_wrist: np.ndarray | None) -> Hand | None:
    """The hand that drives the orb and finger SFX: the one continuing from last frame's wrist,
    else the largest. MediaPipe's list order isn't stable, so without this the orb would jump
    between hands."""
    if not hands:
        return None
    if prev_wrist is None:
        return max(hands, key=lambda h: palm_size(h.lm))
    return min(hands, key=lambda h: float(np.linalg.norm(h.lm[WRIST, :2] - prev_wrist)))
