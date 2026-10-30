# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 lfarabi
"""Per-user data folder: settings, jurus, custom gestures and imported media, saved safely."""
import copy
import filecmp
import json
import os
import re
import shutil
from pathlib import Path

import yaml

APP_NAME = "Meetru"

DEFAULT_SETTINGS = {
    "camera_index": 0, "mic": None, "speaker": None, "meeting_output": "CABLE Input",
    "virtual_camera": True, "language": None, "mirror_preview": True, "skeleton": False,
    "privacy_ack": False,   # the first-run notice (MediaPipe reports usage metrics to Google) was accepted
}


def _sound(n: int, mode: str = "once") -> dict:
    return {"file": f"builtin:finger_{n}", "mode": mode, "volume": 1.0}


# Canonical form the jurus editor emits: no `sound`/`visual` key when there is none.
DEFAULT_JURUS = [
    {"name": "Bola Energi", "enabled": True, "gesture": "OPEN_PALM", "hold": 0.3, "exclusive": False,
     "visual": {"type": "orb"}},
    {"name": "Beep 1", "enabled": True, "gesture": "ONE", "hold": 0.3, "exclusive": False, "sound": _sound(1)},
    {"name": "Beep 2", "enabled": True, "gesture": "TWO", "hold": 0.3, "exclusive": False, "sound": _sound(2)},
    {"name": "Beep 3", "enabled": True, "gesture": "THREE", "hold": 0.3, "exclusive": False, "sound": _sound(3)},
    {"name": "Beep 4", "enabled": True, "gesture": "FOUR", "hold": 0.3, "exclusive": False,
     "sound": _sound(4, "loop")},
    {"name": "Menghilang", "enabled": True, "gesture": "SEAL", "hold": 0.3, "exclusive": True,
     "sound": _sound(3), "visual": {"type": "vanish"}},
    {"name": "WOW", "enabled": True, "gesture": "WOW", "hold": 0.3, "exclusive": True, "sound": _sound(2),
     "visual": {"type": "text", "text": "WOW!", "color": "#ffd400", "position": "top", "size": 0.25,
                "duration": 2.0}},
]


def default_root() -> Path:
    # Qt is imported here so the rest of this module (and its tests) work without PySide6.
    from PySide6.QtCore import QCoreApplication, QStandardPaths
    QCoreApplication.setApplicationName(APP_NAME)
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))


class Store:
    """Reads and writes the user's files under `root`; a corrupt file is set aside, never lost."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.media = self.root / "media"
        self.media.mkdir(parents=True, exist_ok=True)
        self.background = self.root / "background.png"
        self.problems: list[str] = []  # names of files that were unreadable, for the UI to report

    def _read(self, name: str, parse, kind: type, default):
        path = self.root / name
        if not path.exists():
            return copy.deepcopy(default)
        try:
            with open(path, encoding="utf-8") as f:
                data = parse(f)
            if not isinstance(data, kind):
                raise ValueError(f"{name}: expected {kind.__name__}")
            return data
        except (yaml.YAMLError, ValueError):
            os.replace(path, str(path) + ".broken")
            self.problems.append(name)
            return copy.deepcopy(default)

    def _write(self, name: str, dump, data) -> None:
        # Write beside the target, then swap in, so a crash never leaves a half-written file.
        path = self.root / name
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            dump(data, f)
        os.replace(tmp, path)

    def load_settings(self) -> dict:
        return {**DEFAULT_SETTINGS, **self._read("settings.yaml", yaml.safe_load, dict, {})}

    def save_settings(self, d: dict) -> None:
        self._write("settings.yaml", lambda x, f: yaml.safe_dump(x, f, allow_unicode=True, sort_keys=False), d)

    def load_jurus(self) -> list[dict]:
        return self._read("jurus.yaml", yaml.safe_load, list, DEFAULT_JURUS)

    def save_jurus(self, lst: list[dict]) -> None:
        self._write("jurus.yaml", lambda x, f: yaml.safe_dump(x, f, allow_unicode=True, sort_keys=False), lst)

    def load_gestures(self) -> list[dict]:
        return self._read("gestures.json", json.load, list, [])

    def save_gestures(self, lst: list[dict]) -> None:
        self._write("gestures.json", lambda x, f: json.dump(x, f, ensure_ascii=False, indent=2), lst)

    def import_media(self, src: Path) -> str:
        """Copy a file into media/ under a safe name; the same content is reused, a name clash gets _2, _3..."""
        src = Path(src)
        stem, ext = os.path.splitext(re.sub(r"[^A-Za-z0-9._-]+", "_", src.name))
        target, n = self.media / (stem + ext), 1
        while target.exists() and not filecmp.cmp(src, target, shallow=False):
            n += 1
            target = self.media / f"{stem}_{n}{ext}"
        if not target.exists():
            shutil.copyfile(src, target)
        return f"media/{target.name}"

    def resolve(self, ref: str) -> Path | None:
        """Disk path of a `media/...` reference; builtin sounds and empty refs have none."""
        if ref.startswith("media/"):
            return self.media / Path(ref[len("media/"):]).name  # .name: refs can't climb out of media/
        return None
