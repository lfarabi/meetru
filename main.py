# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 lfarabi
"""Meetru: webcam hand gestures → sounds, orb and vanish effects, out to a virtual camera for Meet/Zoom.
Starts the Qt app; the frame loop is engine.py, the window is ui.py."""
import logging
import sys
from pathlib import Path

import yaml
from PySide6.QtWidgets import QApplication

import i18n
from legal import ensure_privacy_notice
from store import APP_NAME, Store, default_root


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from ui import MainWindow  # here, so importing main never needs the window
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)  # no organization name: it would add a folder level to the data path
    store = Store(default_root())
    cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text(encoding="utf-8"))
    i18n.set_language(store.load_settings()["language"])  # the language is picked at start, not live
    if not ensure_privacy_notice(store):  # asked once, before the camera and MediaPipe start
        return 0
    window = MainWindow(store, cfg)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
