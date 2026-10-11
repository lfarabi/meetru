"""The MeetCam window: live preview, status lights, the main buttons and the Settings tab."""
import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QSizePolicy, QTabWidget, QVBoxLayout, QWidget)

import audio
from engine import Engine
from i18n import t

log = logging.getLogger(__name__)


def save_setting(store, engine, key: str, value):
    """One setting changed: write it to disk, then tell the engine."""
    s = store.load_settings()
    s[key] = value
    store.save_settings(s)
    engine.set_settings(s)


def _devices(host_api: str, kind: str) -> list[str]:
    try:
        return audio.list_devices(host_api, kind)
    except Exception as e:  # no PortAudio, or no such host API: an empty picker, not a crash
        log.warning("Could not list the %s devices: %s", kind, e)
        return []


def _fill(combo: QComboBox, items: list[tuple[str, object]], saved):
    """Fill `combo` with (text, value) items and show `saved`. Saved names may be part of a device name (that is
    how the engine matches them), and a saved device that is unplugged stays listed, so it is not lost silently."""
    for text, value in items:
        combo.addItem(text, value)
    i = combo.findData(saved) if saved is not None else 0
    if i < 0 and isinstance(saved, str):
        i = next((n for n in range(combo.count())
                  if isinstance(combo.itemData(n), str) and saved.lower() in combo.itemData(n).lower()), -1)
    if i < 0:
        combo.addItem(saved if isinstance(saved, str) else t("camera_n", n=saved), saved)
        i = combo.count() - 1
    combo.setCurrentIndex(max(i, 0))


class SettingsTab(QWidget):
    """Devices, language and preview options. Every change is saved at once and sent to the engine."""

    def __init__(self, store, engine, cfg: dict):
        super().__init__()
        self.store, self.engine = store, engine
        s, host = store.load_settings(), cfg["audio"]["host_api"]
        cameras = [d.description() for d in QMediaDevices.videoInputs()] or [t("camera_n", n=i) for i in range(4)]
        inputs, outputs = _devices(host, "input"), _devices(host, "output")
        default = [(t("device_default"), None)]
        # The meeting output is VB-CABLE's playback side unless the user chose another.
        meeting = s["meeting_output"] or next((n for n in outputs if "CABLE Input" in n), None)

        self.camera, self.mic, self.speaker, self.meeting, self.language = (QComboBox() for _ in range(5))
        self.vcam, self.mirror = QCheckBox(t("set_vcam")), QCheckBox(t("set_mirror"))
        self.language_note = QLabel("")
        combos = {"camera_index": (self.camera, [(c, i) for i, c in enumerate(cameras)]),
                  "mic": (self.mic, default + [(n, n) for n in inputs]),
                  "speaker": (self.speaker, default + [(n, n) for n in outputs]),
                  "meeting_output": (self.meeting, [(n, n) for n in outputs]),
                  "language": (self.language, [(t("lang_auto"), None), ("Bahasa Indonesia", "id"), ("English", "en")])}
        shown = {**s, "meeting_output": meeting}
        for key, (combo, items) in combos.items():
            # a long device name must not widen the window
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(30)
            _fill(combo, items, shown[key])  # nothing is connected yet, so filling saves nothing
            combo.currentIndexChanged.connect(lambda _, k=key, c=combo: self._changed(k, c.currentData()))
        for key, box in (("virtual_camera", self.vcam), ("mirror_preview", self.mirror)):
            box.setChecked(bool(s[key]))
            box.toggled.connect(lambda on, k=key: self._changed(k, on))

        form = QFormLayout(self)
        for label, widget in (("set_camera", self.camera), ("set_mic", self.mic), ("set_speaker", self.speaker),
                              ("set_meeting_output", self.meeting)):
            form.addRow(t(label), widget)
        form.addRow(self.vcam)
        form.addRow(t("set_language"), self.language)
        form.addRow(self.language_note)
        form.addRow(self.mirror)

    def _changed(self, key: str, value):
        save_setting(self.store, self.engine, key, value)
        if key == "language":
            self.language_note.setText(t("restart_for_language"))


class MainWindow(QMainWindow):
    def __init__(self, store, cfg: dict, engine: Engine | None = None):
        """With no `engine`, makes one and starts it; one passed in is used as it is (tests give an unstarted one)."""
        super().__init__()
        self.setWindowTitle("MeetCam")
        self.store = store
        settings = store.load_settings()
        start = engine is None
        self.engine = engine if engine is not None else Engine(cfg, store, settings)
        self.engine.frame_ready.connect(self.on_frame)
        self.engine.status.connect(self.on_status)

        self.preview = QLabel(alignment=Qt.AlignCenter)
        self.preview.setMinimumSize(480, 270)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # else each frame resizes the window
        self.preview.setStyleSheet("background: black")
        self.message = QLabel(alignment=Qt.AlignCenter, wordWrap=True)
        self.vcam_light, self.meeting_light, self.fps_label = QLabel(), QLabel(), QLabel()
        self.gesture_label = QLabel()
        self.invisible_badge = QLabel(t("badge_invisible"))
        self.invisible_badge.setStyleSheet("background: #6c3fc5; color: white; padding: 1px 8px; border-radius: 8px")
        self.invisible_badge.hide()
        for light in (self.vcam_light, self.meeting_light):
            light.setTextFormat(Qt.RichText)

        lights = QHBoxLayout()
        for w in (self.vcam_light, self.meeting_light, self.fps_label, self.gesture_label):
            lights.addWidget(w)
        lights.addStretch()
        lights.addWidget(self.invisible_badge)

        self.effects = self._button("btn_effects", "Space", lambda on: setattr(self.engine, "effects_on", on), True)
        self.mute = self._button("btn_mute", "M", lambda on: setattr(self.engine, "muted", on))
        self.skeleton = self._button("btn_skeleton", "D", self._set_skeleton, settings["skeleton"])
        self.room = QPushButton(t("btn_room"))
        self.room.clicked.connect(self.engine.capture_room)
        QShortcut(QKeySequence("B"), self, activated=self.room.click)
        QShortcut(QKeySequence("Q"), self, activated=self.close)
        buttons = QHBoxLayout()
        for b in (self.effects, self.mute, self.room, self.skeleton):
            b.setFocusPolicy(Qt.NoFocus)  # so Space is the Effects shortcut, never "press the focused button"
            buttons.addWidget(b)

        self.tabs = QTabWidget()
        self.settings_tab = SettingsTab(store, self.engine, cfg)
        self.tabs.addTab(self.settings_tab, t("tab_settings"))

        root = QWidget()
        column = QVBoxLayout(root)
        column.addWidget(self.preview, 1)
        column.addWidget(self.message)
        column.addLayout(lights)
        column.addLayout(buttons)
        column.addWidget(self.tabs)
        self.setCentralWidget(root)
        self.resize(900, 860)

        self.on_status({})  # the lights start out OFF
        if start:
            self.engine.start()
        if store.problems:  # after the engine, which reads the jurus and gestures files too
            QTimer.singleShot(0, self._report_problems)

    def _button(self, text_key: str, key: str, slot, checked: bool = False) -> QPushButton:
        """A checkable button with its keyboard shortcut; the shortcut clicks the button, so both stay in step."""
        b = QPushButton(t(text_key))
        b.setCheckable(True)
        b.setChecked(checked)
        b.toggled.connect(slot)
        QShortcut(QKeySequence(key), self, activated=b.toggle)
        return b

    def _set_skeleton(self, on: bool):
        self.engine.skeleton = on
        save_setting(self.store, self.engine, "skeleton", on)

    def on_frame(self, image):
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_status(self, st: dict):
        for light, key, on, tip in ((self.vcam_light, "light_vcam", st.get("vcam"), "tip_vcam_off"),
                                    (self.meeting_light, "light_meeting", st.get("meeting"), "tip_meeting_off")):
            light.setText(f'<span style="color: {"#2fb344" if on else "#d63939"}">●</span> '
                          + t(key, state=t("state_on" if on else "state_off")))
            light.setToolTip("" if on else t(tip))
        self.fps_label.setText(t("light_fps", fps=f"{st.get('fps', 0):.0f}"))
        gesture = st.get("gesture", "")
        self.gesture_label.setText("" if gesture == "NO_HAND" else gesture)
        key = st.get("message", "")
        self.message.setText(t(key, **st.get("message_args", {})) if key else "")
        self.invisible_badge.setVisible(bool(st.get("invisible")))

    def problems_text(self) -> str:
        return t("problems_text", files=", ".join(self.store.problems))

    def _report_problems(self):
        box = QMessageBox(QMessageBox.Warning, "MeetCam", self.problems_text(), parent=self)
        box.setWindowModality(Qt.NonModal)  # the window stays usable
        box.setAttribute(Qt.WA_DeleteOnClose)
        box.show()

    def closeEvent(self, event):
        if self.engine.isRunning():
            self.engine.stop()
            self.engine.wait(3000)
        super().closeEvent(event)
