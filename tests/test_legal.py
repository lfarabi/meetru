# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 lfarabi
import pytest
from PySide6.QtWidgets import QDialog, QMessageBox, QPlainTextEdit

import i18n
import legal
from store import Store


@pytest.fixture(autouse=True)
def english():
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def asking(monkeypatch, answer):
    asked = []
    monkeypatch.setattr(legal, "_ask_privacy", lambda: asked.append(1) or answer)
    return asked


def test_the_notice_is_asked_until_accepted_then_remembered(tmp_path, monkeypatch):
    asked = asking(monkeypatch, True)
    assert legal.ensure_privacy_notice(Store(tmp_path))
    assert Store(tmp_path).load_settings()["privacy_ack"] is True
    assert legal.ensure_privacy_notice(Store(tmp_path)) and len(asked) == 1   # not asked again


def test_declining_stops_the_start_and_is_asked_again_next_time(tmp_path, monkeypatch):
    asked = asking(monkeypatch, False)
    store = Store(tmp_path)
    assert not legal.ensure_privacy_notice(store)
    assert store.load_settings()["privacy_ack"] is False
    assert not legal.ensure_privacy_notice(store) and len(asked) == 2


def test_an_unwritable_settings_file_does_not_crash_the_notice(tmp_path, monkeypatch):
    asking(monkeypatch, True)
    store = Store(tmp_path)

    def locked(_):
        raise PermissionError("locked")
    monkeypatch.setattr(store, "save_settings", locked)
    assert legal.ensure_privacy_notice(store)


@pytest.mark.parametrize("lang, go, quit_", [("en", "Continue", "Quit"), ("id", "Lanjutkan", "Keluar")])
def test_the_notice_box_offers_continue_or_quit_and_names_the_source(qapp, monkeypatch, lang, go, quit_):
    i18n.set_language(lang)
    seen = {}
    monkeypatch.setattr(QMessageBox, "exec",
                        lambda self: seen.update(text=self.text(), buttons=[b.text() for b in self.buttons()]))
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self.buttons()[0])
    assert legal._ask_privacy() is True
    assert seen["buttons"] == [go, quit_]
    assert "MediaPipe" in seen["text"] and legal.MEDIAPIPE_PRIVACY_URL in seen["text"]
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self.buttons()[1])
    assert legal._ask_privacy() is False
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)   # closed with the window's X
    assert legal._ask_privacy() is False


@pytest.mark.parametrize("lang, no_warranty", [("en", "ABSOLUTELY NO WARRANTY"), ("id", "TIDAK DISERTAI JAMINAN")])
def test_about_shows_the_notices_the_gpl_asks_for(qapp, lang, no_warranty):
    i18n.set_language(lang)
    text = legal.AboutDialog().text.toPlainText()
    for needle in (no_warranty, "Meetru", "2026", legal.COPYRIGHT_OWNER, "GNU General Public License", legal.GPL_URL,
                   "MediaPipe", "pyvirtualcam"):
        assert needle in text, needle


def test_the_license_button_shows_the_license_file(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(QDialog, "exec", lambda self: shown.append(self.findChild(QPlainTextEdit).toPlainText()))
    legal.AboutDialog().license.click()
    assert shown and shown[0].lstrip().startswith("GNU GENERAL PUBLIC LICENSE") and "Version 3" in shown[0][:200]


def test_without_the_license_file_it_points_to_the_official_text(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(legal, "LICENSE_FILE", tmp_path / "missing")
    seen = {}
    monkeypatch.setattr(QMessageBox, "exec", lambda self: seen.update(text=self.text()))
    legal.show_license()
    assert legal.GPL_URL in seen["text"]
