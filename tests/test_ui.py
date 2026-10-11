import string

import pytest
from conftest import M_CFG, make_engine
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

import i18n
import ui
from i18n import STRINGS, t
from store import Store
from ui import MainWindow, SettingsTab

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


def fake_audio(monkeypatch, devices):
    """The sound card's devices as {"input": [...], "output": [...]}, or an exception for a broken driver."""
    def list_devices(host_api, kind):
        if isinstance(devices, Exception):
            raise devices
        return devices[kind]
    monkeypatch.setattr(ui.audio, "list_devices", list_devices)


def fake_cameras(monkeypatch, names):
    """The cameras Qt reports, or an exception for a failing query."""
    class Camera:
        def __init__(self, name):
            self.name = name

        def description(self):
            return self.name

    class Media:
        @staticmethod
        def videoInputs():
            if isinstance(names, Exception):
                raise names
            return [Camera(n) for n in names]
    monkeypatch.setattr(ui, "QMediaDevices", Media)


def texts(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def settings_tab(tmp_path, **saved):
    store = Store(tmp_path)
    if saved:
        store.save_settings({**store.load_settings(), **saved})
    return SettingsTab(store, make_engine(store), M_CFG)


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


# --- the pickers, on made-up devices ----------------------------------------------------------
CABLE = "CABLE Input (VB-Audio Virtual Cable)"


def test_a_saved_part_of_a_device_name_selects_the_whole_device(tmp_path, monkeypatch, qapp):
    fake_audio(monkeypatch, {"input": ["Mic"], "output": ["Speakers", CABLE]})
    tab = settings_tab(tmp_path)  # the default meeting_output is "CABLE Input"
    assert (tab.meeting.currentText(), tab.meeting.currentData()) == (CABLE, CABLE)
    assert not (tmp_path / "settings.yaml").exists()  # showing it is not saving it


def test_no_saved_meeting_output_selects_the_first_cable_input(tmp_path, monkeypatch, qapp):
    fake_audio(monkeypatch, {"input": [], "output": ["Speakers", "Headset", CABLE, "CABLE Input 2"]})
    assert settings_tab(tmp_path, meeting_output=None).meeting.currentData() == CABLE


def test_a_failing_device_query_gives_empty_pickers(tmp_path, monkeypatch, qapp):
    fake_audio(monkeypatch, OSError("no PortAudio"))
    tab = settings_tab(tmp_path)
    assert texts(tab.mic) == texts(tab.speaker) == [t("device_default")]
    assert texts(tab.meeting) == ["CABLE Input"]  # only the saved one, still shown


@pytest.mark.parametrize("cameras", [[], RuntimeError("no camera service")])
def test_no_cameras_listed_gives_camera_0_to_3(tmp_path, monkeypatch, qapp, cameras):
    fake_cameras(monkeypatch, cameras)
    assert texts(settings_tab(tmp_path).camera) == [f"Camera {i}" for i in range(4)]


def test_cameras_are_listed_by_name_and_index(tmp_path, monkeypatch, qapp):
    fake_cameras(monkeypatch, ["FHD Camera", "Other"])
    tab = settings_tab(tmp_path, camera_index=1)
    assert texts(tab.camera) == ["FHD Camera", "Other"] and tab.camera.currentData() == 1


def test_every_string_has_the_same_fields_in_both_languages():
    def fields(text):
        return {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}
    assert {k: fields(v) for k, v in STRINGS["id"].items()} == {k: fields(v) for k, v in STRINGS["en"].items()}


def test_a_null_skeleton_setting_does_not_crash_the_start(tmp_path, qapp):
    store = Store(tmp_path)
    store.save_settings({**store.load_settings(), "skeleton": None})
    assert not MainWindow(store, M_CFG, make_engine(store)).skeleton.isChecked()


def test_the_tabs_sit_to_the_right_of_the_preview_on_a_laptop_screen(window, qapp):
    window.show()
    qapp.processEvents()
    left, right = (w.mapTo(window, QPoint(0, 0)).x() for w in (window.preview, window.tabs))
    assert right >= left + window.preview.width()
    assert window.width() <= 1280 and window.height() <= 700  # fits 1366x768 with the taskbar


def test_a_broken_file_is_reported_once_the_window_is_up(tmp_path, qapp):
    (tmp_path / "settings.yaml").write_text("{ not yaml", encoding="utf-8")
    store = Store(tmp_path)
    w = MainWindow(store, M_CFG, make_engine(store))
    assert store.problems == ["settings.yaml"]
    assert "settings.yaml" in w.problems_text()
