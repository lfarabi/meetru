import os
from pathlib import Path

import pytest
import yaml

# Qt tests must not need a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

M_CFG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8"))


_app = None  # the one QApplication; a Qt object nobody refers to is destroyed


def ensure_qapp():
    global _app
    from PySide6.QtWidgets import QApplication  # lazy: tests that don't need Qt don't pay for it
    _app = QApplication.instance() or QApplication([])
    return _app


@pytest.fixture(scope="session")
def qapp():
    return ensure_qapp()


class FakeTracker:
    """Stands in for HandTracker: sees whatever is in `hands`."""

    def __init__(self, hands=None):
        self.hands, self.closed = hands or [], False

    def detect(self, frame, t_ms):
        return self.hands

    def close(self):
        self.closed = True


def make_engine(store, settings=None, tracker=None):
    """An unstarted Engine on fake devices: no camera, no sound card, no virtual camera. Its default jurus
    include a text visual, which needs a QApplication, so this makes sure there is one."""
    ensure_qapp()
    from audio import AudioEngine
    from engine import Engine
    from store import DEFAULT_SETTINGS
    cfg = {**M_CFG, "audio": {**M_CFG["audio"], "enabled": False}}  # a reopened AudioEngine stays silent too
    return Engine(cfg, store, DEFAULT_SETTINGS if settings is None else settings,
                  tracker=tracker or FakeTracker(), audio=AudioEngine(cfg["audio"]))
