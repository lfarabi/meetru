import os
from pathlib import Path

import pytest
import yaml

# Qt tests must not need a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

M_CFG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication  # lazy: tests that don't need Qt don't pay for it
    return QApplication.instance() or QApplication([])


class FakeTracker:
    """Stands in for HandTracker: sees whatever is in `hands`."""

    def __init__(self, hands=None):
        self.hands = hands or []

    def detect(self, frame, t_ms):
        return self.hands

    def close(self):
        pass


def make_engine(store, settings=None, tracker=None):
    """An unstarted Engine on fake devices: no camera, no sound card, no virtual camera. Its jurus
    include a text visual, so the test needs the `qapp` fixture."""
    from audio import AudioEngine
    from engine import Engine
    from store import DEFAULT_SETTINGS
    cfg = {**M_CFG, "audio": {**M_CFG["audio"], "enabled": False}}  # a reopened AudioEngine stays silent too
    return Engine(cfg, store, DEFAULT_SETTINGS if settings is None else settings,
                  tracker=tracker or FakeTracker(), audio=AudioEngine(cfg["audio"]))
