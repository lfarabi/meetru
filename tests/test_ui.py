import string
from pathlib import Path

import pytest
from conftest import M_CFG, make_engine
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

import i18n
import ui
from gestures import BUILTIN_GESTURES
from i18n import STRINGS, t
from store import DEFAULT_JURUS, Store
from ui import GestureTab, JurusEditor, JurusTab, MainWindow, SettingsTab

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


@pytest.mark.parametrize("lang", ["en", "id"])
def test_the_tabs_sit_to_the_right_of_the_preview_on_a_laptop_screen(tmp_path, qapp, lang):
    i18n.set_language(lang)
    store = Store(tmp_path)
    window = MainWindow(store, M_CFG, make_engine(store))
    window.show()
    qapp.processEvents()
    left, right = (w.mapTo(window, QPoint(0, 0)).x() for w in (window.preview, window.tabs))
    assert right >= left + window.preview.width()
    assert window.width() <= 1366 and window.height() <= 700  # fits 1366x768 with the taskbar
    # the smallest the window can be made, with the widest texts the lights and the badge show
    for on in (True, False):
        window.on_status({**STATUS, "vcam": on, "meeting": on, "gesture": "RAISED_FIST", "invisible": True})
        hint = window.minimumSizeHint()
        assert hint.width() <= 1366 and hint.height() <= 680, (lang, on, hint)


def test_a_broken_file_is_reported_once_the_window_is_up(tmp_path, qapp):
    (tmp_path / "settings.yaml").write_text("{ not yaml", encoding="utf-8")
    store = Store(tmp_path)
    w = MainWindow(store, M_CFG, make_engine(store))
    assert store.problems == ["settings.yaml"]
    assert "settings.yaml" in w.problems_text()


# --- the Jurus tab and its editor --------------------------------------------------------------
def editor(tmp_path, jurus=None, taken=(), **kw):
    return JurusEditor(Store(tmp_path), jurus, set(taken), **kw)


def choose_file(monkeypatch, path):
    """Make the file dialog 'choose' this path; returns the filters it was opened with."""
    seen = []

    def get_open_file_name(parent, title, directory, filter_):
        seen.append(filter_)
        return str(path), filter_
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", staticmethod(get_open_file_name))
    return seen


def warnings(monkeypatch):
    """Collect the texts of the warnings shown instead of opening a message box."""
    shown = []
    monkeypatch.setattr(ui.QMessageBox, "warning", staticmethod(lambda parent, title, text: shown.append(text)))
    return shown


def source(tmp_path, name):
    (tmp_path / "in").mkdir(exist_ok=True)
    (tmp_path / "in" / name).write_bytes(b"not really media")
    return tmp_path / "in" / name


def jurus_tab(tmp_path, jurus=None):
    store = Store(tmp_path)
    if jurus is not None:
        store.save_jurus(jurus)
    return JurusTab(store, make_engine(store))


def pick_in(combo, data):
    combo.setCurrentIndex(combo.findData(data))


def press(monkeypatch, text):
    """Make a message box 'press' the button with this text (None: it is closed, as by Escape) instead of waiting.
    Returns what each box showed."""
    shown = []

    def fake_exec(self):
        shown.append({"text": self.text(), "buttons": {b.text() for b in self.buttons()},
                      "default": self.defaultButton().text()})
        for b in self.buttons():
            if b.text() == text:
                b.click()
        return 0
    monkeypatch.setattr(ui.QMessageBox, "exec", fake_exec)
    return shown


def test_editor_round_trips_a_jurus(tmp_path, qapp):
    j = DEFAULT_JURUS[-1]                    # WOW text jurus
    assert JurusEditor(Store(tmp_path), j, set()).value() == j


@pytest.mark.parametrize("j", DEFAULT_JURUS, ids=[j["name"] for j in DEFAULT_JURUS])
def test_editor_round_trips_every_default_jurus(tmp_path, qapp, j):
    assert editor(tmp_path, j).value() == j


def test_editor_round_trips_the_other_visual_kinds(tmp_path, qapp):
    base = {"name": "X", "enabled": False, "gesture": "FIST", "hold": 0.5, "exclusive": True}
    for extra in ({"visual": {"type": "image", "file": "media/a.png", "position": "hand", "size": 0.5, "duration": "held"}},
                  {"visual": {"type": "video", "file": "media/a.mp4", "position": "full", "size": 1.0, "duration": 3.5},
                   "sound": {"file": "media/s.wav", "mode": "once", "volume": 0.55}},
                  {"visual": {"type": "orb", "file": "media/o.mp4"},
                   "orb_sounds": {"activation": "media/a.wav", "blast": "media/b.wav"}},
                  {"visual": {"type": "vanish", "file": "media/smoke.mp4"}}):
        j = {**base, **extra}
        assert editor(tmp_path, j).value() == j


def test_orb_and_vanish_are_greyed_out_when_taken(tmp_path, qapp):
    e = JurusEditor(Store(tmp_path), None, {"orb"})
    assert not e.visual_type.model().item(e.visual_type.findData("orb")).isEnabled()
    assert e.visual_type.model().item(e.visual_type.findData("vanish")).isEnabled()
    e = JurusEditor(Store(tmp_path), None, {"orb", "vanish"})
    assert not e.visual_type.model().item(e.visual_type.findData("vanish")).isEnabled()


def test_jurus_tab_flags_a_missing_file(tmp_path, qapp):
    s = Store(tmp_path)
    s.save_jurus([{**DEFAULT_JURUS[1], "sound": {"file": "media/gone.mp3", "mode": "once", "volume": 1.0}}])
    tab = JurusTab(s, make_engine(s))
    assert "⚠" in tab.list.item(0).text()
    assert tab.list.item(0).toolTip() == t("problem_sound")


def test_the_default_jurus_are_listed_checked_and_without_a_warning(tmp_path, qapp):
    tab = jurus_tab(tmp_path)
    rows = [tab.list.item(i) for i in range(tab.list.count())]
    assert [r.text().split(" — ")[0] for r in rows] == [j["name"] for j in DEFAULT_JURUS]
    assert all("⚠" not in r.text() and r.checkState() == Qt.Checked and r.toolTip() == "" for r in rows)
    assert tab.engine._pending is None  # showing the list does not reload the engine


def test_unchecking_a_jurus_saves_it_and_reloads_the_engine(tmp_path, qapp):
    tab = jurus_tab(tmp_path)
    tab.list.item(1).setCheckState(Qt.Unchecked)
    assert [j["enabled"] for j in Store(tmp_path).load_jurus()] == [True, False] + [True] * 5
    assert [j.live for j in tab.engine._pending.rules.jurus][:3] == [True, False, True]


def test_a_non_dict_entry_in_jurus_yaml_does_not_break_the_tab(tmp_path, qapp):
    tab = jurus_tab(tmp_path, ["junk", DEFAULT_JURUS[1]])
    assert tab.list.count() == 2 and "⚠" in tab.list.item(0).text() and "⚠" not in tab.list.item(1).text()


def test_delete_asks_first_in_the_current_language(tmp_path, monkeypatch, qapp):
    i18n.set_language("id")
    tab = jurus_tab(tmp_path)
    tab.list.setCurrentRow(1)
    shown = press(monkeypatch, t("btn_cancel"))
    tab.delete_btn.click()
    assert tab.list.count() == len(DEFAULT_JURUS) and tab.engine._pending is None
    # the box's own buttons, in Indonesian (not Qt's English Yes/No), and Enter does not delete
    assert shown == [{"text": t("confirm_delete", name="Beep 1"), "buttons": {"Hapus", "Batal"}, "default": "Batal"}]
    press(monkeypatch, None)
    tab.delete_btn.click()
    assert tab.list.count() == len(DEFAULT_JURUS) and tab.engine._pending is None
    press(monkeypatch, t("btn_delete"))
    tab.delete_btn.click()
    assert [j["name"] for j in Store(tmp_path).load_jurus()] == [j["name"] for j in DEFAULT_JURUS if j["name"] != "Beep 1"]
    assert len(tab.engine._pending.rules.jurus) == len(DEFAULT_JURUS) - 1 and tab.list.count() == len(DEFAULT_JURUS) - 1


def test_a_fresh_tab_has_the_first_jurus_selected(tmp_path, qapp):
    tab = jurus_tab(tmp_path)
    assert tab.list.currentRow() == 0
    assert all(b.isEnabled() for b in (tab.add_btn, tab.edit_btn, tab.delete_btn, tab.try_btn))


def test_edit_delete_and_try_need_a_selected_jurus(tmp_path, monkeypatch, qapp):
    tab = jurus_tab(tmp_path, [])
    needing_one = (tab.edit_btn, tab.delete_btn, tab.try_btn)
    assert tab.list.currentRow() == -1 and tab.add_btn.isEnabled() and not any(b.isEnabled() for b in needing_one)
    added = type("E", (), {"__init__": lambda self, *a, **k: None, "exec": lambda self: QDialog.Accepted,
                           "deleteLater": lambda self: None, "value": lambda self: DEFAULT_JURUS[1]})
    monkeypatch.setattr(ui, "JurusEditor", added)
    tab.add_btn.click()
    assert tab.list.currentRow() == 0 and all(b.isEnabled() for b in needing_one)
    press(monkeypatch, t("btn_delete"))
    tab.delete_btn.click()
    assert tab.list.count() == 0 and not any(b.isEnabled() for b in needing_one)


def test_try_sends_the_selected_jurus_to_the_engine(tmp_path, qapp):
    tab = jurus_tab(tmp_path)
    tab.list.setCurrentRow(2)
    tab.try_btn.click()
    assert queued(tab.engine) == [("try_jurus", ("Beep 2",))]


def test_add_and_edit_open_the_editor_with_what_the_other_jurus_use(tmp_path, monkeypatch, qapp):
    tab, opened = jurus_tab(tmp_path), []

    class Stub:
        def __init__(self, store, jurus, taken, engine=None, names=(), parent=None):
            opened.append((jurus and jurus["name"], taken, set(names)))
            self.jurus = jurus

        def exec(self):
            return QDialog.Accepted

        def deleteLater(self):
            pass

        def value(self):
            return {**(self.jurus or DEFAULT_JURUS[1]), "name": "Changed"}
    monkeypatch.setattr(ui, "JurusEditor", Stub)
    tab.list.setCurrentRow(0)  # Orb (the orb): only the vanish is used by the others
    tab.edit_btn.click()
    others = {j["name"] for j in DEFAULT_JURUS} - {"Orb"}
    assert opened[-1] == ("Orb", {"vanish"}, others)
    assert Store(tmp_path).load_jurus()[0]["name"] == "Changed" and tab.list.item(0).text().startswith("Changed")
    tab.add_btn.click()
    assert opened[-1] == (None, {"orb", "vanish"}, others | {"Changed"})
    assert Store(tmp_path).load_jurus()[-1]["name"] == "Changed" and tab.list.count() == len(DEFAULT_JURUS) + 1


def test_a_cancelled_editor_changes_nothing(tmp_path, monkeypatch, qapp):
    tab = jurus_tab(tmp_path)
    cancelled = type("E", (), {"exec": lambda self: QDialog.Rejected, "deleteLater": lambda self: None})
    monkeypatch.setattr(ui, "JurusEditor", lambda *a, **k: cancelled())
    tab.add_btn.click()
    assert tab.list.count() == len(DEFAULT_JURUS) and tab.engine._pending is None


def test_a_name_must_be_given_and_unique_and_the_dialog_stays_open(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    e = editor(tmp_path, None, names={"WOW"})
    e.accept()
    e.name.setText("  WOW ")
    e.accept()
    assert e.result() == QDialog.Rejected and shown == [t("err_name_empty"), t("err_name_taken", name="WOW")]
    e.name.setText("Other")
    e.accept()
    assert e.result() == QDialog.Accepted and e.value()["name"] == "Other"


def test_a_picture_needs_a_file_and_a_text_needs_words(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    e = editor(tmp_path)
    e.name.setText("P")
    pick_in(e.visual_type, "image")
    e.accept()
    pick_in(e.visual_type, "text")
    e.accept()
    assert e.result() == QDialog.Rejected and shown == [t("err_need_file"), t("err_need_text")]


def test_a_loop_sound_cannot_be_exclusive(tmp_path, qapp):
    e = editor(tmp_path, {**DEFAULT_JURUS[-1], "sound": {"file": "builtin:finger_1", "mode": "once", "volume": 1.0}})
    assert e.exclusive.isChecked() and e.exclusive.isEnabled()
    pick_in(e.sound_mode, "loop")
    assert not e.exclusive.isChecked() and not e.exclusive.isEnabled()
    assert e.value()["exclusive"] is False and e.value()["sound"]["mode"] == "loop"
    pick_in(e.sound_mode, "once")
    assert e.exclusive.isEnabled()


def test_the_fields_shown_follow_the_visual_type(tmp_path, qapp):
    e = editor(tmp_path)
    parts = (e.visual_file, e.text, e.color_btn, e.position, e.orb_files["blast"], e.sound_src)

    def shown():
        return [w.isVisibleTo(e) for w in parts]
    assert shown() == [False, False, False, False, False, True]
    for kind, expected in (("image", [True, False, False, True, False, True]),
                           ("text", [False, True, True, True, False, True]),
                           ("orb", [True, False, False, False, True, False]),
                           ("vanish", [True, False, False, False, False, True])):
        pick_in(e.visual_type, kind)
        assert shown() == expected, kind


def test_an_orb_has_no_sound_but_the_orb_sounds(tmp_path, qapp):
    e = editor(tmp_path, {**DEFAULT_JURUS[0], "sound": {"file": "builtin:finger_1", "mode": "once", "volume": 1.0}})
    assert e.value() == DEFAULT_JURUS[0]  # the sound an orb cannot play is not kept


def test_picking_a_sound_imports_it_and_uses_the_sound_filter(tmp_path, monkeypatch, qapp):
    seen = choose_file(monkeypatch, source(tmp_path, "ding dong.wav"))
    e = editor(tmp_path)
    pick_in(e.sound_src, ui.CHOOSE_FILE)
    assert "*.mp3 *.wav *.ogg *.flac" in seen[0]
    assert (tmp_path / "media" / "ding_dong.wav").exists()
    assert e.sound_src.currentText() == "ding_dong.wav"
    assert e.value()["sound"] == {"file": "media/ding_dong.wav", "mode": "once", "volume": 1.0}
    assert editor(tmp_path, e.value()).value()["sound"] == e.value()["sound"]  # and it is shown again when reopened


def test_cancelling_the_file_dialog_keeps_the_previous_sound(tmp_path, monkeypatch, qapp):
    choose_file(monkeypatch, "")
    e = editor(tmp_path, DEFAULT_JURUS[1])
    pick_in(e.sound_src, ui.CHOOSE_FILE)
    assert e.value()["sound"]["file"] == "builtin:finger_1"


def test_a_file_that_cannot_be_copied_is_reported_not_raised(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    choose_file(monkeypatch, tmp_path / "in" / "nope.wav")
    e = editor(tmp_path)
    pick_in(e.sound_src, ui.CHOOSE_FILE)
    assert len(shown) == 1 and "sound" not in e.value()


def test_the_visual_pickers_use_the_image_and_video_filters(tmp_path, monkeypatch, qapp):
    seen = choose_file(monkeypatch, source(tmp_path, "a.png"))
    e = editor(tmp_path)
    for kind, patterns in (("image", "*.png *.jpg *.jpeg *.gif"), ("video", "*.mp4 *.mov *.webm *.avi"),
                           ("orb", "*.mp4 *.mov *.webm *.avi"), ("vanish", "*.mp4 *.mov *.webm *.avi")):
        pick_in(e.visual_type, kind)
        e.visual_file.button.click()
        assert patterns in seen[-1], kind
    assert e.value()["visual"] == {"type": "vanish", "file": "media/a.png"}
    pick_in(e.visual_type, "image")  # the file belongs to the type it was picked for
    assert e.visual_file.ref is None


def test_the_orb_sound_pickers(tmp_path, monkeypatch, qapp):
    seen = choose_file(monkeypatch, source(tmp_path, "boom.ogg"))
    e = editor(tmp_path, {"name": "O", "enabled": True, "gesture": "OPEN_PALM", "hold": 0.3, "exclusive": False,
                          "visual": {"type": "orb"}})
    e.orb_files["blast"].button.click()
    assert "*.mp3 *.wav *.ogg *.flac" in seen[0]
    assert e.value()["orb_sounds"] == {"blast": "media/boom.ogg"}
    e.orb_files["blast"].clear.click()
    assert "orb_sounds" not in e.value()


def test_the_colour_button_stores_the_colour_as_hex(tmp_path, monkeypatch, qapp):
    e = editor(tmp_path, DEFAULT_JURUS[-1])
    asked = []
    monkeypatch.setattr(ui.QColorDialog, "getColor", staticmethod(lambda *a, **k: asked.append(a) or QColor(255, 0, 128)))
    e.color_btn.click()
    assert e.value()["visual"]["color"] == "#ff0080"
    assert asked[0][1:] == (e, t("ed_color"))  # parent and a title in the app's language
    monkeypatch.setattr(ui.QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor()))  # cancelled: invalid
    e.color_btn.click()
    assert e.value()["visual"]["color"] == "#ff0080"


def test_recorded_gestures_are_listed_and_a_missing_one_is_kept_and_marked(tmp_path, qapp):
    store = Store(tmp_path)
    store.save_gestures([{"name": "Wave", "hands": 1, "mean": [0.0] * 210, "radius": 1.0, "sensitivity": 1.0}])
    j = {**DEFAULT_JURUS[1], "gesture": "rec:Wave"}
    e = JurusEditor(store, j, set())
    assert e.gesture.currentText() == "🖐 Wave" and e.value()["gesture"] == "rec:Wave"
    assert e.gesture.itemData(e.gesture.count() - 1) == "rec:Wave"
    gone = JurusEditor(store, {**j, "gesture": "rec:Gone"}, set())
    assert gone.gesture.currentText() == "⚠ rec:Gone" and gone.value()["gesture"] == "rec:Gone"


MEETING = {"enabled": True, "mic_device": "x", "output_device": "y", "mic_volume": 1.0, "sfx_volume": 0.8,
           "max_latency_ms": 60}


def test_the_preview_of_a_builtin_is_heard_on_the_monitor_and_not_in_the_meeting(tmp_path, qapp):
    store = Store(tmp_path)
    engine = make_engine(store)
    engine.audio.add_meeting_mixer(MEETING)  # no devices needed
    e = JurusEditor(store, {**DEFAULT_JURUS[1], "sound": {"file": "builtin:finger_3", "mode": "once", "volume": 0.5}},
                    set(), engine=engine)
    e.preview_btn.click()
    assert engine.audio.mixer._queue.qsize() == 1 and engine.audio.meeting_mixer._queue.qsize() == 0
    assert engine.audio.mixer._queue.get_nowait() == ("finger_3", 0.5, False)


def test_the_preview_never_raises(tmp_path, qapp):
    store = Store(tmp_path)
    engine = make_engine(store)
    e = JurusEditor(store, DEFAULT_JURUS[1], set(), engine=engine)
    engine.audio.mixer.play = lambda *a: 1 / 0
    e.preview_btn.click()  # logged, not raised
    silent = editor(tmp_path, DEFAULT_JURUS[1])
    silent.preview_btn.click()  # no engine: nothing to play through, no crash


def test_the_preview_plays_a_media_file_with_the_media_player(tmp_path, monkeypatch, qapp):
    log = []

    class Player:
        def __init__(self, parent=None):
            log.append("player")

        def setAudioOutput(self, out):
            log.append("output")

        def setSource(self, url):
            log.append(Path(url.toLocalFile()))

        def play(self):
            log.append("play")

        def stop(self):
            log.append("stop")

    class Output:
        def __init__(self, parent=None):
            pass

        def setVolume(self, v):
            log.append(v)
    monkeypatch.setattr(ui, "QMediaPlayer", Player)
    monkeypatch.setattr(ui, "QAudioOutput", Output)
    e = editor(tmp_path, {**DEFAULT_JURUS[1], "sound": {"file": "media/ding.wav", "mode": "once", "volume": 0.25}})
    e.preview_btn.click()
    assert log == ["player", "output", 0.25, tmp_path / "media" / "ding.wav", "play"]
    e.reject()
    assert log[-1] == "stop"


def test_the_jurus_tab_comes_first_and_is_selected(window):
    assert window.tabs.widget(0) is window.jurus_tab and window.tabs.currentWidget() is window.jurus_tab
    assert window.tabs.tabText(0) == t("tab_jurus") and window.tabs.widget(2) is window.settings_tab


def test_the_language_note_wraps(window):
    assert window.settings_tab.language_note.wordWrap()


# --- the Gesture tab ---------------------------------------------------------------------------
HAND = {"name": "", "hands": 1, "mean": [0.0] * 210, "radius": 0.05, "sensitivity": 1.5}   # what the engine emits


def gesture(name, **over):
    return {**HAND, "name": name, **over}


def gesture_tab(tmp_path, gestures=(), jurus=None):
    store = Store(tmp_path)
    store.save_gestures(list(gestures))
    if jurus is not None:
        store.save_jurus(jurus)
    return GestureTab(store, make_engine(store))


def buttons(tab):
    return tab.record_btn, tab.rerecord_btn, tab.rename_btn, tab.delete_btn


def ask(monkeypatch, name):
    """Make the name dialog 'answer' with this name (None: cancelled) instead of opening; returns what it was asked."""
    asked = []

    def _ask_name(self, title, current=""):
        asked.append((title, current))
        return name
    monkeypatch.setattr(GestureTab, "_ask_name", _ask_name)
    return asked


def window_with(tmp_path, gestures, jurus):
    store = Store(tmp_path)
    store.save_gestures(gestures)
    store.save_jurus(jurus)
    return MainWindow(store, M_CFG, make_engine(store))


def test_recorded_gesture_is_saved_and_offered_in_the_editor(tmp_path, qapp):
    s = Store(tmp_path)
    tab = GestureTab(s, make_engine(s))
    tab.pending_name = "peace"
    tab.on_recorded({"name": "", "hands": 1, "mean": [0.0] * 210, "radius": 0.05, "sensitivity": 1.5})
    assert s.load_gestures()[0]["name"] == "peace"
    assert JurusEditor(s, None, set()).gesture.findData("rec:peace") >= 0


def test_rename_updates_jurus_that_use_it(tmp_path, monkeypatch, qapp):
    other = {**DEFAULT_JURUS[2], "gesture": "rec:keep"}
    tab = gesture_tab(tmp_path, [gesture("peace"), gesture("keep")],
                      [{**DEFAULT_JURUS[1], "gesture": "rec:peace"}, other])
    changed, asked = [], ask(monkeypatch, "victory")
    tab.gestures_changed.connect(lambda: changed.append(1))
    tab.recorded_list.setCurrentRow(0)
    tab.rename_btn.click()
    store = Store(tmp_path)
    assert asked == [(t("gest_rename_title"), "peace")] and changed == [1]
    assert [g["name"] for g in store.load_gestures()] == ["victory", "keep"]
    assert [j["gesture"] for j in store.load_jurus()] == ["rec:victory", "rec:keep"]
    assert tab.recorded_list.currentItem().text() == "🖐 victory"


@pytest.mark.parametrize("key", ["no_hand", "unsteady"])
def test_failed_recording_shows_a_message_and_saves_nothing(tmp_path, monkeypatch, qapp, key):
    shown, titles = [], []
    monkeypatch.setattr(ui.QMessageBox, "warning",
                        staticmethod(lambda parent, title, text: (titles.append(title), shown.append(text))))
    tab = gesture_tab(tmp_path)
    changed = []
    tab.gestures_changed.connect(lambda: changed.append(1))
    tab.pending_name = "peace"
    tab.record_btn.setEnabled(False)
    tab.on_recorded(key)
    assert shown == [t("record_" + key)] and titles == [t("record_failed_title")]
    assert t("record_" + key) != "record_" + key   # a real sentence, in both languages
    assert Store(tmp_path).load_gestures() == [] and changed == [] and tab.pending_name is None
    assert tab.record_btn.isEnabled()


def test_record_asks_for_a_name_then_records_and_the_buttons_wait_for_the_result(tmp_path, monkeypatch, qapp):
    tab = gesture_tab(tmp_path, [gesture("old")])
    asked = ask(monkeypatch, "peace")
    assert all(b.isEnabled() for b in buttons(tab))
    tab.record_btn.click()
    assert asked == [(t("gest_name_title"), "")] and tab.pending_name == "peace"
    assert queued(tab.engine) == [("record", ())] and not any(b.isEnabled() for b in buttons(tab))
    tab.on_recorded(dict(HAND))
    assert tab.pending_name is None and all(b.isEnabled() for b in buttons(tab))
    assert [g["name"] for g in Store(tmp_path).load_gestures()] == ["old", "peace"]
    assert tab.recorded_list.currentItem().text() == "🖐 peace"   # the new one is selected


@pytest.mark.parametrize("name, problem", [("", "err_gest_empty"), ("   ", "err_gest_empty"),
                                           ("old", "err_gest_taken"), ("a:b", "err_gest_colon")])
def test_a_bad_name_is_refused_with_a_warning_and_nothing_is_recorded(tmp_path, monkeypatch, qapp, name, problem):
    shown = warnings(monkeypatch)
    tab = gesture_tab(tmp_path, [gesture("old")])
    ask(monkeypatch, name)
    tab.record_btn.click()
    assert shown == [t(problem, name=name.strip())] and queued(tab.engine) == []
    assert tab.pending_name is None and all(b.isEnabled() for b in buttons(tab))


def test_a_cancelled_name_dialog_does_nothing(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    tab = gesture_tab(tmp_path, [gesture("old")])
    ask(monkeypatch, None)
    tab.record_btn.click()
    tab.rename_btn.click()
    assert shown == [] and queued(tab.engine) == [] and tab.pending_name is None
    assert [g["name"] for g in Store(tmp_path).load_gestures()] == ["old"]


def test_renaming_to_a_taken_name_is_refused_and_to_the_same_name_is_a_no_op(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    tab = gesture_tab(tmp_path, [gesture("a"), gesture("b")])
    changed = []
    tab.gestures_changed.connect(lambda: changed.append(1))
    tab.recorded_list.setCurrentRow(0)
    ask(monkeypatch, "b")
    tab.rename_btn.click()
    ask(monkeypatch, "a")
    tab.rename_btn.click()
    assert shown == [t("err_gest_taken", name="b")] and changed == []
    assert [g["name"] for g in Store(tmp_path).load_gestures()] == ["a", "b"]


def test_the_name_dialog_is_in_the_app_language(tmp_path, monkeypatch, qapp):
    i18n.set_language("id")
    tab = gesture_tab(tmp_path)
    seen = []

    def fake_exec(self):
        seen.append((self.windowTitle(), self.labelText(), self.textValue(), self.okButtonText(),
                     self.cancelButtonText()))
        self.setTextValue("  Damai  ")
        return QDialog.Accepted
    monkeypatch.setattr(ui.QInputDialog, "exec", fake_exec)
    assert tab._new_name(t("gest_name_title"), "Lama") == "Damai"   # trimmed
    assert seen == [(t("gest_name_title"), t("gest_name_label"), "Lama", "OK", "Batal")]
    monkeypatch.setattr(ui.QInputDialog, "exec", lambda self: QDialog.Rejected)
    assert tab._new_name(t("gest_name_title")) is None


def test_rerecord_replaces_the_template_and_keeps_the_sensitivity(tmp_path, qapp):
    tab = gesture_tab(tmp_path, [gesture("peace", sensitivity=2.2), gesture("other")])
    tab.recorded_list.setCurrentRow(0)
    tab.rerecord_btn.click()
    assert tab.pending_name == "peace" and queued(tab.engine) == [("record", ())]
    tab.on_recorded({**HAND, "mean": [0.5] * 210, "radius": 0.07, "sensitivity": 1.5})
    saved = Store(tmp_path).load_gestures()
    assert [g["name"] for g in saved] == ["peace", "other"]
    assert saved[0] == {"name": "peace", "hands": 1, "mean": [0.5] * 210, "radius": 0.07, "sensitivity": 2.2}


def test_a_result_nobody_asked_for_is_ignored(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    tab = gesture_tab(tmp_path)
    tab.on_recorded(dict(HAND))
    tab.on_recorded("no_hand")
    assert Store(tmp_path).load_gestures() == [] and shown == []


def test_the_sensitivity_slider_saves_after_a_pause(tmp_path, qapp):
    tab = gesture_tab(tmp_path, [gesture("peace")])
    changed = []
    tab.gestures_changed.connect(lambda: changed.append(1))
    assert (tab.sens.minimum(), tab.sens.maximum(), tab.sens.value()) == (8, 30, 15)
    assert tab.sens_value.text() == "1.5×"
    tab.sens.setValue(22)
    tab.sens.setValue(23)
    assert tab.sens_value.text() == "2.3×" and tab._timer.isSingleShot() and tab._timer.interval() == 400
    assert tab._timer.isActive() and changed == []
    assert Store(tmp_path).load_gestures()[0]["sensitivity"] == 1.5   # not yet: saving rebuilds the media
    tab._timer.timeout.emit()
    assert Store(tmp_path).load_gestures()[0]["sensitivity"] == 2.3 and changed == [1]


def test_picking_another_gesture_loads_its_sensitivity_without_saving(tmp_path, qapp):
    tab = gesture_tab(tmp_path, [gesture("a", sensitivity=1.0), gesture("b", sensitivity=2.5)])
    assert tab.sens.value() == 10 and tab.sens.isEnabled()
    tab.recorded_list.setCurrentRow(1)
    assert tab.sens.value() == 25 and tab.sens_value.text() == "2.5×" and not tab._timer.isActive()
    assert [g["sensitivity"] for g in Store(tmp_path).load_gestures()] == [1.0, 2.5]


def test_the_live_score_bar_follows_the_selected_gesture(tmp_path, qapp):
    tab = gesture_tab(tmp_path, [gesture("peace"), gesture("other")])
    bar = tab.score_bar
    assert (bar.minimum(), bar.maximum(), bar.value(), bar.format()) == (0, 100, 0, "—")
    tab.on_status({"scores": {"peace": 0.5, "other": 0.9}})
    assert (bar.value(), bar.format()) == (50, t("score_match"))
    tab.on_status({"scores": {"peace": 1.7}})
    assert bar.value() == 100
    tab.on_status({"scores": {"peace": -0.4}})   # outside the template's tolerance
    assert (bar.value(), bar.format()) == (0, "%p%")
    for scores in ({"peace": None}, {"other": 0.9}, {}):   # the hands don't fit, or not known to the engine yet
        tab.on_status({"scores": scores})
        assert (bar.value(), bar.format()) == (0, "—")
    tab.on_status({})
    assert (bar.value(), bar.format()) == (0, "—")
    tab.on_status({"scores": {"peace": 0.5, "other": 0.9}})
    tab.recorded_list.setCurrentRow(1)
    assert (bar.value(), bar.format()) == (0, "—")   # a new selection starts blank until the next frame
    tab.on_status({"scores": {"peace": 0.5, "other": 0.9}})
    assert bar.value() == 90


def test_with_no_recording_the_bar_and_slider_idle_and_the_edit_buttons_are_off(tmp_path, qapp):
    tab = gesture_tab(tmp_path)
    assert tab.record_btn.isEnabled() and not any(b.isEnabled() for b in buttons(tab)[1:])
    assert not tab.sens.isEnabled() and tab.sens_value.text() == "" and tab.recorded_list.currentRow() == -1
    tab.on_status({"scores": {"peace": 0.9}})
    assert tab.score_bar.value() == 0


def test_the_lists_show_the_built_in_gestures_and_the_recordings(tmp_path, qapp):
    tab = gesture_tab(tmp_path, [gesture("peace"), gesture("clap", hands=2, mean=[0.0] * 421)])
    assert [tab.builtin_list.item(i).text() for i in range(tab.builtin_list.count())] == \
        [t("g_" + g) for g in BUILTIN_GESTURES]
    assert tab.builtin_list.selectionMode() == tab.builtin_list.SelectionMode.NoSelection   # read-only
    assert [tab.recorded_list.item(i).text() for i in range(tab.recorded_list.count())] == \
        ["🖐 peace", "🖐 clap" + t("gest_hands2")]
    assert t("gest_hands2") == " (2 hands)"
    assert all(b.isEnabled() for b in buttons(tab))   # the first recording is selected


def test_a_gestures_file_with_junk_in_it_does_not_break_the_tab_and_keeps_the_junk(tmp_path, monkeypatch, qapp):
    tab = gesture_tab(tmp_path, ["junk", {"name": "half"}, gesture("peace")])
    assert tab.recorded_list.count() == 1
    ask(monkeypatch, "victory")
    tab.rename_btn.click()
    saved = Store(tmp_path).load_gestures()
    assert saved[:2] == ["junk", {"name": "half"}] and saved[2]["name"] == "victory"


def test_delete_asks_first_then_removes_the_gesture_and_the_jurus_stays_flagged(tmp_path, monkeypatch, qapp):
    i18n.set_language("id")
    store = Store(tmp_path)
    w = window_with(tmp_path, [gesture("peace")], [{**DEFAULT_JURUS[1], "gesture": "rec:peace"}])
    tab, jurus = w.gesture_tab, w.jurus_tab
    assert "⚠" not in jurus.list.item(0).text()
    shown = press(monkeypatch, t("btn_cancel"))
    tab.delete_btn.click()
    assert shown == [{"text": t("confirm_delete_gesture", name="peace"), "buttons": {"Hapus", "Batal"},
                      "default": "Batal"}]
    assert [g["name"] for g in store.load_gestures()] == ["peace"]   # cancelled
    press(monkeypatch, None)   # closed with Escape
    tab.delete_btn.click()
    assert [g["name"] for g in store.load_gestures()] == ["peace"]
    press(monkeypatch, t("btn_delete"))
    tab.delete_btn.click()
    assert store.load_gestures() == [] and tab.recorded_list.count() == 0 and not tab.delete_btn.isEnabled()
    assert store.load_jurus()[0]["gesture"] == "rec:peace" and "⚠" in jurus.list.item(0).text()


def test_the_gesture_tab_is_wired_into_the_window(tmp_path, monkeypatch, qapp):
    w = window_with(tmp_path, [gesture("peace")], [{**DEFAULT_JURUS[1], "gesture": "rec:peace"}, DEFAULT_JURUS[2]])
    assert w.tabs.widget(1) is w.gesture_tab and w.tabs.tabText(1) == t("tab_gestures")
    w.engine.status.emit({**STATUS, "scores": {"peace": 0.25}})
    assert w.gesture_tab.score_bar.value() == 25
    ask(monkeypatch, "victory")
    w.gesture_tab.rename_btn.click()   # the jurus tab must not write its old list back over the renamed jurus
    store = Store(tmp_path)
    assert [j["gesture"] for j in store.load_jurus()] == ["rec:victory", DEFAULT_JURUS[2]["gesture"]]
    assert w.jurus_tab.items[0]["gesture"] == "rec:victory" and "⚠" not in w.jurus_tab.list.item(0).text()
    assert w.engine._pending is not None   # and the engine was handed the new setup
    ask(monkeypatch, "clap")
    w.gesture_tab.record_btn.click()
    w.engine.recorded.emit({**HAND, "mean": [0.1] * 210})
    assert [g["name"] for g in store.load_gestures()] == ["victory", "clap"]


def test_the_jurus_tab_and_the_gesture_tab_ask_alike_before_deleting(tmp_path, monkeypatch, qapp):
    deleted = []
    monkeypatch.setattr(ui.QMessageBox, "deleteLater", lambda self: deleted.append(self.text()))
    press(monkeypatch, t("btn_cancel"))
    jurus_tab(tmp_path).delete_btn.click()
    gesture_tab(tmp_path / "g", [gesture("peace")]).delete_btn.click()
    assert deleted == [t("confirm_delete", name="Orb"), t("confirm_delete_gesture", name="peace")]


def failing_disk(monkeypatch, store, *names):
    """Make these writes of the store fail as a locked disk does; returns the warning the user should get."""
    def refuse(data):
        raise PermissionError(13, "Permission denied")
    for name in names:
        monkeypatch.setattr(store, name, refuse)
    return t("err_gest_save", error=PermissionError(13, "Permission denied"))


def test_a_recording_that_cannot_be_saved_is_reported_and_the_buttons_come_back(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    tab = gesture_tab(tmp_path, [gesture("old")])
    message = failing_disk(monkeypatch, tab.store, "save_gestures")
    changed = []
    tab.gestures_changed.connect(lambda: changed.append(1))
    tab.pending_name = "peace"
    tab._sync_buttons()
    assert not tab.record_btn.isEnabled()
    tab.on_recorded(dict(HAND))   # no crash
    assert shown == [message] and "Permission denied" in message
    assert tab.pending_name is None and all(b.isEnabled() for b in buttons(tab)) and changed == []
    assert [g["name"] for g in tab.items] == ["old"] and tab.recorded_list.count() == 1   # memory = disk


def test_a_rename_that_cannot_be_saved_changes_nothing_and_says_so(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    tab = gesture_tab(tmp_path, [gesture("peace")], [{**DEFAULT_JURUS[1], "gesture": "rec:peace"}])
    message = failing_disk(monkeypatch, tab.store, "save_gestures")
    ask(monkeypatch, "victory")
    tab.rename_btn.click()
    assert shown == [message] and tab.items[0]["name"] == "peace" and tab.recorded_list.item(0).text() == "🖐 peace"
    assert Store(tmp_path).load_jurus()[0]["gesture"] == "rec:peace"   # the jurus were not touched either


def test_a_rename_whose_jurus_cannot_be_saved_says_so_and_the_rename_stands(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    tab = gesture_tab(tmp_path, [gesture("peace")], [{**DEFAULT_JURUS[1], "gesture": "rec:peace"}])
    message = failing_disk(monkeypatch, tab.store, "save_jurus")
    changed = []
    tab.gestures_changed.connect(lambda: changed.append(1))
    ask(monkeypatch, "victory")
    tab.rename_btn.click()
    # the jurus keep the old name, so the Jurus tab (told by gestures_changed) marks them with the warning sign
    assert shown == [message] and changed == [1]
    assert [g["name"] for g in Store(tmp_path).load_gestures()] == ["victory"] and tab.items[0]["name"] == "victory"
    assert tab.recorded_list.item(0).text() == "🖐 victory"


def test_a_delete_that_cannot_be_saved_keeps_the_gesture_and_says_so(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    tab = gesture_tab(tmp_path, [gesture("peace")])
    message = failing_disk(monkeypatch, tab.store, "save_gestures")
    press(monkeypatch, t("btn_delete"))
    tab.delete_btn.click()
    assert shown == [message] and tab.recorded_list.count() == 1 and [g["name"] for g in tab.items] == ["peace"]


def test_a_slider_change_that_cannot_be_saved_says_so(tmp_path, monkeypatch, qapp):
    shown = warnings(monkeypatch)
    tab = gesture_tab(tmp_path, [gesture("peace")])
    message = failing_disk(monkeypatch, tab.store, "save_gestures")
    tab.sens.setValue(20)
    tab._timer.timeout.emit()
    assert shown == [message]


@pytest.mark.parametrize("big", [1e308, 99, True])
def test_a_sensitivity_beyond_the_slider_does_not_crash_the_tab(tmp_path, qapp, big):
    tab = gesture_tab(tmp_path, [gesture("peace", sensitivity=big)])
    assert 8 <= tab.sens.value() <= 30 and tab.sens_value.text().endswith("×")
    tab.on_status({"scores": {"peace": float("nan")}})
    assert (tab.score_bar.value(), tab.score_bar.format()) == (0, "—")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -5, 0])
def test_an_unusable_sensitivity_drops_the_recording_and_recording_it_again_heals_its_jurus(tmp_path, monkeypatch,
                                                                                            qapp, bad):
    tab = gesture_tab(tmp_path, [gesture("peace", sensitivity=bad)], [{**DEFAULT_JURUS[1], "gesture": "rec:peace"}])
    assert tab.recorded_list.count() == 0 and not tab.sens.isEnabled()
    assert JurusTab(tab.store, tab.engine).list.item(0).text().startswith("⚠")
    ask(monkeypatch, "peace")                                          # the name is free again
    tab.record_btn.click()
    tab.on_recorded(dict(HAND))
    assert tab.recorded_list.count() == 1 and tab.sens.value() == 15
    assert "⚠" not in JurusTab(tab.store, tab.engine).list.item(0).text()
