import os

# Qt tests must not need a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
