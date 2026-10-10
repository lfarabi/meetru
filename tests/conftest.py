import os

import pytest

# Qt tests must not need a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication  # lazy: tests that don't need Qt don't pay for it
    return QApplication.instance() or QApplication([])
