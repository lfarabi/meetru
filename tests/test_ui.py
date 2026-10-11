import pytest
from conftest import M_CFG, make_engine
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

import i18n
from i18n import STRINGS, t
from store import Store
from ui import MainWindow

STATUS = {"fps": 25.0, "vcam": True, "meeting": True, "gesture": "NO_HAND", "message": "", "message_args": {},
          "scores": {}, "invisible": False, "effects_on": True, "muted": False}


@pytest.fixture(autouse=True)
def english():
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def queued(engine):
    """The commands the UI has sent the (unstarted) engine so far."""
    out = []
    while not engine._cmds.empty():
        out.append(engine._cmds.get_nowait())
    return out


@pytest.fixture
def window(tmp_path, qapp):
    store = Store(tmp_path)
    return MainWindow(store, M_CFG, make_engine(store))


# --- i18n -------------------------------------------------------------------------
def test_every_string_exists_in_both_languages():
    assert STRINGS["id"].keys() == STRINGS["en"].keys()


def test_a_missing_key_returns_the_key_and_fields_are_filled_in():
    assert t("no_such_key") == "no_such_key"
    assert "3" in t("room_countdown", seconds=3)


def test_language_none_follows_the_system_locale(monkeypatch):
    class Locale:  # a Windows PC set to Indonesian
        @staticmethod
        def system():
            return type("L", (), {"name": staticmethod(lambda: "id_ID")})()
    monkeypatch.setattr(i18n, "QLocale", Locale)
    i18n.set_language(None)
    assert t("tab_settings") == STRINGS["id"]["tab_settings"]
    i18n.set_language("fr")  # unknown → English
    assert t("tab_settings") == STRINGS["en"]["tab_settings"]


# --- the window -------------------------------------------------------------------------
def test_main_window_builds_and_settings_save(tmp_path, qapp):
    w = MainWindow(Store(tmp_path), M_CFG, make_engine(Store(tmp_path)))
    w.settings_tab.mirror.setChecked(False)
    assert Store(tmp_path).load_settings()["mirror_preview"] is False


def test_a_setting_reaches_the_engine_and_the_other_settings_survive(tmp_path, window):
    window.settings_tab.vcam.setChecked(False)
    window.settings_tab.mirror.setChecked(False)
    saved = Store(tmp_path).load_settings()
    assert saved["virtual_camera"] is False and saved["mirror_preview"] is False
    assert queued(window.engine)[-1] == ("set_settings", (saved,))


def test_the_saved_devices_are_restored_without_writing_anything(tmp_path, qapp):
    store = Store(tmp_path)
    store.save_settings({**store.load_settings(), "mic": "Gone Mic", "camera_index": 7, "language": "id"})
    before = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    w = MainWindow(store, M_CFG, make_engine(store))  # the saved mic and camera are not in the lists
    tab = w.settings_tab
    assert (tab.mic.currentData(), tab.camera.currentData(), tab.language.currentData()) == ("Gone Mic", 7, "id")
    assert (tmp_path / "settings.yaml").read_text(encoding="utf-8") == before


def test_changing_the_language_saves_it_and_says_to_restart(tmp_path, window):
    tab = window.settings_tab
    tab.language.setCurrentIndex(tab.language.findData("id"))
    assert Store(tmp_path).load_settings()["language"] == "id"
    assert tab.language_note.text() == t("restart_for_language")


def test_on_status_shows_the_message_translated(window):
    window.on_status({**STATUS, "message": "camera_failed"})
    assert window.message.text() == STRINGS["en"]["camera_failed"]
    window.on_status({**STATUS, "message": "room_countdown", "message_args": {"seconds": 4}})
    assert "4" in window.message.text()
    window.on_status(STATUS)
    assert window.message.text() == ""


def test_on_status_lights_and_tooltips(window):
    window.on_status({**STATUS, "vcam": False, "gesture": "OPEN_PALM", "invisible": True})
    assert "OFF" in window.vcam_light.text() and "OBS" in window.vcam_light.toolTip()
    assert window.meeting_light.toolTip() == ""
    assert "25" in window.fps_label.text()
    assert window.gesture_label.text() == "OPEN_PALM"
    assert not window.invisible_badge.isHidden()
    window.on_status(STATUS)
    assert window.gesture_label.text() == "" and window.invisible_badge.isHidden()


def test_buttons_and_shortcuts_drive_the_engine(tmp_path, window):
    e = window.engine
    window.effects.click()
    assert e.effects_on is False
    window.mute.click()
    assert e.muted is True
    window.skeleton.click()
    assert e.skeleton is True and Store(tmp_path).load_settings()["skeleton"] is True
    window.skeleton.setChecked(False)
    assert e.skeleton is False and Store(tmp_path).load_settings()["skeleton"] is False
    queued(e)
    window.room.click()
    assert queued(e) == [("capture_room", ())]


def test_the_keyboard_shortcuts(window):
    e = window.engine
    window.show()
    QTest.qWaitForWindowActive(window)
    for key in (Qt.Key_Space, Qt.Key_M, Qt.Key_D):
        QTest.keyClick(window, key)
    assert (e.effects_on, e.muted, e.skeleton) == (False, True, True)
    assert (window.effects.isChecked(), window.mute.isChecked(), window.skeleton.isChecked()) == (False, True, True)
    queued(e)
    QTest.keyClick(window, Qt.Key_B)
    assert queued(e) == [("capture_room", ())]
    QTest.keyClick(window, Qt.Key_Q)
    assert not window.isVisible()


def test_a_broken_file_is_reported_once_the_window_is_up(tmp_path, qapp):
    (tmp_path / "settings.yaml").write_text("{ not yaml", encoding="utf-8")
    store = Store(tmp_path)
    w = MainWindow(store, M_CFG, make_engine(store))
    assert store.problems == ["settings.yaml"]
    assert "settings.yaml" in w.problems_text()
