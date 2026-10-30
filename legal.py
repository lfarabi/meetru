# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 lfarabi
"""What the window owes its users legally: the notice shown before the first start (MediaPipe sends usage metrics to
Google) and the About box with the notices the GPL asks a program to display (copyright, no warranty, the license and
how to read it)."""
import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QPlainTextEdit, QTextBrowser, QVBoxLayout

from i18n import t
from store import APP_NAME

log = logging.getLogger(__name__)

COPYRIGHT_YEAR, COPYRIGHT_OWNER = 2026, "lfarabi"
LICENSE_FILE = Path(__file__).with_name("LICENSE")
GPL_URL = "https://www.gnu.org/licenses/gpl-3.0.html"
MEDIAPIPE_PRIVACY_URL = "https://developers.google.com/edge/mediapipe/solutions/tasks"


def _ask_privacy() -> bool:
    """The notice before the first start; True = Continue, anything else (Quit, Esc, the window's X) = no."""
    box = QMessageBox(QMessageBox.Information, APP_NAME, t("privacy_text", name=APP_NAME, url=MEDIAPIPE_PRIVACY_URL))
    box.setTextFormat(Qt.RichText)
    go = box.addButton(t("privacy_continue"), QMessageBox.AcceptRole)
    box.addButton(t("privacy_quit"), QMessageBox.RejectRole)
    box.exec()
    accepted = box.clickedButton() is go
    box.deleteLater()
    return accepted


def ensure_privacy_notice(store) -> bool:
    """Asks until the notice has been accepted once; False = declined, so the camera must not start.

    ponytail: a yes/no flag; make it a version number if the notice's text ever changes in a way that needs a new yes."""
    settings = store.load_settings()
    if settings.get("privacy_ack"):
        return True
    if not _ask_privacy():
        return False
    try:
        store.save_settings({**settings, "privacy_ack": True})
    except OSError as e:  # an unwritable data folder: it is simply asked again at the next start
        log.warning("Could not remember that the privacy notice was accepted: %s", e)
    return True


def license_text() -> str | None:
    try:
        return LICENSE_FILE.read_text(encoding="utf-8")
    except OSError:
        return None


def show_license(parent=None):
    text = license_text()
    if text is None:  # a copy without the LICENSE file next to it: point at the official text
        box = QMessageBox(QMessageBox.Information, t("btn_license"), t("license_missing", url=GPL_URL), parent=parent)
        box.setTextFormat(Qt.RichText)
        box.exec()
        box.deleteLater()
        return
    dialog = QDialog(parent)
    dialog.setWindowTitle(t("btn_license"))
    view = QPlainTextEdit(text)
    view.setReadOnly(True)
    view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    buttons = QDialogButtonBox()
    buttons.addButton(t("btn_close"), QDialogButtonBox.RejectRole)
    buttons.rejected.connect(dialog.reject)
    layout = QVBoxLayout(dialog)
    layout.addWidget(view)
    layout.addWidget(buttons)
    dialog.resize(680, 520)
    dialog.exec()
    dialog.deleteLater()


class AboutDialog(QDialog):
    """Copyright, no warranty, the license (and a button to read it), the libraries used, the privacy note."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("about_title", name=APP_NAME))
        self.text = QTextBrowser(openExternalLinks=True)
        self.text.setHtml(t("about_html", name=APP_NAME, year=COPYRIGHT_YEAR, owner=COPYRIGHT_OWNER,
                            gpl=GPL_URL, privacy=MEDIAPIPE_PRIVACY_URL))
        buttons = QDialogButtonBox()
        self.license = buttons.addButton(t("btn_license"), QDialogButtonBox.ActionRole)  # does not close the box
        buttons.addButton(t("btn_close"), QDialogButtonBox.RejectRole)
        self.license.clicked.connect(lambda: show_license(self))
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)
        layout.addWidget(buttons)
        self.resize(560, 540)
