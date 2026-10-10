"""Image, video and text visuals for a jurus: loaders that turn each into a `jutsu.Clip`, where to
put it on the frame, and a player that fades it in and out. Positions are named as seen in the
mirrored preview; the output frame (what the meeting sees) is unmirrored."""
from pathlib import Path

import numpy as np
from PIL import Image

import jutsu
from jutsu import Clip, draw_clip

POSITIONS = ("hand", "full", "top_left", "top", "top_right", "left", "center", "right",
             "bottom_left", "bottom", "bottom_right")
MARGIN = 0.04       # gap to the frame edge for the grid positions, as a fraction of frame height
RELEASE_FADE = 0.2  # seconds a held visual takes to fade out once the gesture is let go


def image_clip(path: Path) -> Clip:
    """A picture or animated GIF/PNG/WebP, alpha kept. Animations play at their mean frame rate."""
    frames, durations = [], []
    with Image.open(path) as im:
        for i in range(getattr(im, "n_frames", 1)):
            im.seek(i)
            rgba = np.asarray(im.convert("RGBA"))
            frames.append((np.ascontiguousarray(rgba[..., 2::-1]), rgba[..., 3].copy()))  # RGB → BGR
            durations.append(im.info.get("duration", 100))
    h, w = frames[0][1].shape
    fps = 1000 / (sum(durations) / len(durations) or 100)  # a 0 ms delay means "default" (100 ms)
    return Clip(frames, fps, 0, (w / 2, h / 2), (0, 0), w)


def text_clip(text: str, color: str, px: int = 160) -> Clip:
    """`text` in bold `color` with a dark outline, one frame. Needs a running QGuiApplication."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen

    fill = QColor(color)
    if not fill.isValid():
        raise ValueError(f"not a colour: {color!r}")
    font = QFont()
    font.setPixelSize(px)
    font.setBold(True)
    path = QPainterPath()
    path.addText(0, 0, font, text)
    outline = max(2, px // 16)
    box = path.boundingRect().adjusted(-outline, -outline, outline, outline)
    img = QImage(int(box.width()) + 1, int(box.height()) + 1, QImage.Format.Format_ARGB32)
    img.fill(0)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(-box.topLeft())
    painter.strokePath(path, QPen(QColor(20, 20, 20), 2 * outline, Qt.PenStyle.SolidLine,
                                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.fillPath(path, fill)  # over the stroke, so the outline shows only outside the letters
    painter.end()
    w, h = img.width(), img.height()
    bgra = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4)
    return Clip([(bgra[..., :3].copy(), bgra[..., 3].copy())], 30.0, 0, (w / 2, h / 2), (0, 0), w)


def visual_clip(visual: dict, path: Path | None) -> Clip:
    """The clip for a jurus's image / video / text visual; raises if it cannot be made."""
    kind = visual["type"]
    if kind == "text":
        return text_clip(visual["text"], visual.get("color", "#ffffff"))
    if kind not in ("image", "video"):
        raise ValueError(f"no clip for a {kind!r} visual")
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"{kind} file not found: {path}")
    return image_clip(path) if kind == "image" else jutsu.load_clip(path)


def place(clip: Clip, position: str, size: float, frame_w: int, frame_h: int,
          palm: tuple[float, float, float] | None, preview: bool) -> tuple[int, int, float] | None:
    """Where to draw `clip`: (x0, y0, scale), for the mirrored preview or the unmirrored output.

    `size` is a fraction of frame height (of palm length for "hand"; ignored for "full").
    `palm` is (x, y, palm_size) in output pixels; "hand" without one gives None. An unknown
    position is "center"."""
    ch, cw = clip.frames[0][0].shape[:2]
    if position == "full":  # the clip's own framing, stretched to the frame width
        scale = frame_w / clip.source_width
        w = max(1, int(cw * scale))
        x0, y0, flip = int(clip.origin[0] * scale), int(clip.origin[1] * scale), preview
    elif position == "hand":
        if palm is None:
            return None
        scale = size * palm[2] / ch
        w, h = max(1, int(cw * scale)), max(1, int(ch * scale))
        x0, y0, flip = round(palm[0] - w / 2), round(palm[1] - h / 2), preview
    else:  # 3x3 grid, worked out as seen in the preview
        scale = size * frame_h / ch
        w, h = max(1, int(cw * scale)), max(1, int(ch * scale))
        m = round(MARGIN * frame_h)
        parts = position.split("_") if position in POSITIONS else ["center"]
        col = 0 if "left" in parts else 2 if "right" in parts else 1
        row = 0 if "top" in parts else 2 if "bottom" in parts else 1
        x0 = (m, (frame_w - w) // 2, frame_w - m - w)[col]
        y0 = (m, (frame_h - h) // 2, frame_h - m - h)[row]
        flip = not preview
    return (frame_w - x0 - w if flip else x0), y0, scale


class OverlayPlayer:
    """Shows one jurus's visual: `duration` "held" (until `release`, then a short fade) or a number
    of seconds. The clip loops while it is up."""

    def __init__(self, clip: Clip, visual: dict):
        self.clip, self.visual = clip, visual
        self._start = self._end = self._released = None

    @property
    def busy(self) -> bool:
        return self._start is not None

    def start(self, t: float, seconds: float | None = None):
        """`seconds` overrides the visual's duration (the editor's Try button)."""
        d = self.visual.get("duration") if seconds is None else seconds
        self._start, self._released = t, None
        self._end = t + d if isinstance(d, (int, float)) else None

    def release(self, t: float):
        if self.busy and self._end is None and self._released is None:
            self._released = t

    def update(self, t: float):
        if self.busy and ((self._end is not None and t >= self._end)
                          or (self._released is not None and t - self._released >= RELEASE_FADE)):
            self._start = None

    def draw(self, frame: np.ndarray, t: float, palm: tuple[float, float, float] | None, preview: bool):
        if not self.busy:
            return
        v = self.visual
        at = place(self.clip, v.get("position", "center"), v.get("size", 0.4), frame.shape[1], frame.shape[0], palm, preview)
        if at is None:
            return
        i = int((t - self._start) * self.clip.fps) % len(self.clip.frames)
        gain = 1.0 if self._released is None else max(0.0, 1 - (t - self._released) / RELEASE_FADE)
        draw_clip(frame, self.clip, i, *at, gain)
