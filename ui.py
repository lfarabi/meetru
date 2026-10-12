"""The MeetCam window: live preview, status lights, the main buttons, the Jurus tab (with its editor), the Gesture
tab and the Settings tab."""
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QKeySequence, QPixmap, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QSlider, QTabWidget, QVBoxLayout, QWidget)

import audio
from engine import Engine
from gestures import BUILTIN_GESTURES
from i18n import t
from overlay import POSITIONS
from poses import DEFAULT_SENSITIVITY
from rules import ORB_SOUNDS, _number, build, valid_templates

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


def _cameras() -> list[str]:
    try:
        return [d.description() for d in QMediaDevices.videoInputs()]
    except Exception as e:  # the camera service is missing: the fallback names, not a crash
        log.warning("Could not list the cameras: %s", e)
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
        cameras = _cameras() or [t("camera_n", n=i) for i in range(4)]
        inputs, outputs = _devices(host, "input"), _devices(host, "output")
        default = [(t("device_default"), None)]
        # The meeting output is VB-CABLE's playback side unless the user chose another.
        meeting = s["meeting_output"] or next((n for n in outputs if "CABLE Input" in n), None)

        self.camera, self.mic, self.speaker, self.meeting, self.language = (QComboBox() for _ in range(5))
        self.vcam, self.mirror = QCheckBox(t("set_vcam")), QCheckBox(t("set_mirror"))
        self.language_note = QLabel("", wordWrap=True)
        combos = {"camera_index": (self.camera, [(c, i) for i, c in enumerate(cameras)]),
                  "mic": (self.mic, default + [(n, n) for n in inputs]),
                  "speaker": (self.speaker, default + [(n, n) for n in outputs]),
                  "meeting_output": (self.meeting, [(n, n) for n in outputs]),
                  "language": (self.language, [(t("lang_auto"), None), ("Bahasa Indonesia", "id"), ("English", "en")])}
        shown = {**s, "meeting_output": meeting}
        for key, (combo, items) in combos.items():
            # a long device name must not widen the window
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(24)
            _fill(combo, items, shown[key])  # nothing is connected yet, so filling saves nothing
            combo.currentIndexChanged.connect(lambda _, k=key, c=combo: self._changed(k, c.currentData()))
        for key, box in (("virtual_camera", self.vcam), ("mirror_preview", self.mirror)):
            box.setChecked(bool(s[key]))
            box.toggled.connect(lambda on, k=key: self._changed(k, on))

        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)  # labels above the pickers: the column stays narrow
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


SOUND_FILES = ("filter_sound", "*.mp3 *.wav *.ogg *.flac")   # (text key for the dialog's filter, its patterns)
IMAGE_FILES = ("filter_image", "*.png *.jpg *.jpeg *.gif")
VIDEO_FILES = ("filter_video", "*.mp4 *.mov *.webm *.avi")
VISUALS = ("none", "image", "video", "text", "orb", "vanish")
FILE_VISUALS = ("image", "video", "orb", "vanish")           # the visuals that take a file (the last two optionally)
SHAPED_VISUALS = ("image", "video", "text")                  # the visuals with a position, size and duration
CHOOSE_FILE = "choose-file"                                  # the sound picker's "File..." entry; not a real reference


def _dict(x) -> dict:
    """A hand-edited jurus.yaml may hold anything where a mapping belongs."""
    return x if isinstance(x, dict) else {}


def pick_file(parent, store, kind: tuple[str, str]) -> str | None:
    """Ask for a media file and copy it into the store: its `media/...` reference, or None if the user cancelled
    or it could not be copied."""
    label, patterns = kind
    path, _ = QFileDialog.getOpenFileName(parent, t("pick_title"), "", f"{t(label)} ({patterns})")
    if not path:
        return None
    try:
        return store.import_media(path)
    except OSError as e:
        QMessageBox.warning(parent, "MeetCam", t("err_import", error=e))
        return None


def _confirm_delete(parent, text: str) -> bool:
    """Ask whether to delete; Enter means no."""
    # not QMessageBox.question: its Yes/No buttons would be in English whatever the language
    box = QMessageBox(QMessageBox.Question, "MeetCam", text, parent=parent)
    delete = box.addButton(t("btn_delete"), QMessageBox.AcceptRole)
    box.setDefaultButton(box.addButton(t("btn_cancel"), QMessageBox.RejectRole))
    box.exec()
    confirmed = box.clickedButton() is delete
    box.deleteLater()
    return confirmed


def gesture_label(gesture) -> str:
    gesture = str(gesture or "")
    if gesture in BUILTIN_GESTURES:
        return t("g_" + gesture)
    return "🖐 " + gesture[len("rec:"):] if gesture.startswith("rec:") else gesture


class FileField(QWidget):
    """A button that picks a media file (copied into the store) and shows its name; ✕ forgets it."""

    def __init__(self, store, kind: tuple[str, str], empty: str):
        super().__init__()
        self.store, self.kind, self.empty, self.ref = store, kind, empty, None
        self.button, self.clear = QPushButton(), QPushButton("✕")
        self.button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)  # a long name is cut, not widening the dialog
        self.clear.setToolTip(t("clear_file"))
        self.clear.setFixedWidth(28)
        self.button.clicked.connect(lambda: self._pick())
        self.clear.clicked.connect(lambda: self.set_ref(None))
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.button, 1)
        row.addWidget(self.clear)
        self.set_ref(None)

    def configure(self, kind: tuple[str, str], empty: str):
        """Now for another kind of file: the one chosen before is forgotten."""
        self.kind, self.empty = kind, empty
        self.set_ref(None)

    def set_ref(self, ref: str | None):
        self.ref = ref or None
        self.button.setText(Path(ref).name if ref else self.empty)
        self.clear.setVisible(bool(ref))

    def _pick(self):
        if ref := pick_file(self, self.store, self.kind):
            self.set_ref(ref)


class JurusEditor(QDialog):
    """Make or change one jurus: its gesture, an optional sound and an optional visual. `taken` holds the
    one-of-a-kind visuals ("orb", "vanish") that the other jurus use, `names` their names. `engine` is only
    for the sound preview."""

    def __init__(self, store, jurus: dict | None, taken: set[str], engine=None, names=(), parent=None):
        super().__init__(parent)
        self.store, self.engine, self.names = store, engine, set(names)
        self._player = self._out = None   # the media preview, made on first use and kept alive here
        j = _dict(jurus)
        sound, visual, orb_sounds = _dict(j.get("sound")), _dict(j.get("visual")), _dict(j.get("orb_sounds"))
        self._enabled = bool(j.get("enabled", True))   # switched in the list, not here
        self.setWindowTitle(t("editor_edit" if jurus else "editor_new"))
        self.setMinimumWidth(420)

        self.name = QLineEdit(str(j.get("name", "")))
        self.gesture = QComboBox()
        for g in BUILTIN_GESTURES:
            self.gesture.addItem(t("g_" + g), g)
        for template in valid_templates(store.load_gestures()):
            self.gesture.addItem("🖐 " + template["name"], "rec:" + template["name"])
        saved = str(j.get("gesture", BUILTIN_GESTURES[0]))
        if self.gesture.findData(saved) < 0:   # a deleted recording: shown and kept, not silently replaced
            self.gesture.addItem("⚠ " + saved, saved)
        self.gesture.setCurrentIndex(self.gesture.findData(saved))

        self.sound_src = QComboBox()
        self.sound_src.addItem(t("choice_none"), "")
        for sound_name in audio.BUILTIN_SOUNDS:
            self.sound_src.addItem(sound_name, "builtin:" + sound_name)
        self.sound_src.addItem(t("sound_file_item"), CHOOSE_FILE)
        self._sound_prev = ""
        self.preview_btn = QPushButton("▶")
        self.preview_btn.setToolTip(t("play_preview"))
        self.preview_btn.setFixedWidth(32)
        self.sound_row = self._row(self.sound_src, self.preview_btn)
        self.sound_mode = QComboBox()
        self.sound_mode.addItem(t("mode_once"), "once")
        self.sound_mode.addItem(t("mode_loop"), "loop")
        self.volume, self.volume_label = QSlider(Qt.Horizontal), QLabel()
        self.volume.setRange(0, 100)
        self.volume_row = self._row(self.volume, self.volume_label)

        self.visual_type = QComboBox()
        for kind in VISUALS:
            self.visual_type.addItem(t("choice_none" if kind == "none" else "v_" + kind), kind)
            if kind in taken:   # there can be only one of these
                self.visual_type.model().item(self.visual_type.count() - 1).setEnabled(False)
        self.visual_file = FileField(store, IMAGE_FILES, "")
        self.text = QLineEdit()
        self.color = "#ffffff"
        self.color_btn = QPushButton()
        self.position = QComboBox()
        for p in POSITIONS:
            self.position.addItem(t("pos_" + p), p)
        self.visual_size = QDoubleSpinBox()
        self.visual_size.setRange(0.05, 3.0)
        self.visual_size.setSingleStep(0.05)
        self.visual_size.setDecimals(2)
        self.held, self.seconds = QCheckBox(t("dur_held")), QDoubleSpinBox()
        self.seconds.setRange(0.1, 60.0)
        self.seconds.setSingleStep(0.5)
        self.seconds.setDecimals(2)
        self.seconds.setSuffix(" s")
        self.duration_row = self._row(self.held, self.seconds)
        self.orb_files = {k: FileField(store, SOUND_FILES, t("device_default")) for k in ORB_SOUNDS}

        self.hold = QDoubleSpinBox()
        self.hold.setRange(0.0, 3.0)
        self.hold.setSingleStep(0.1)
        self.hold.setDecimals(2)
        self.hold.setSuffix(" s")
        self.exclusive = QCheckBox(t("ed_exclusive"))
        hint = QLabel(t("exclusive_hint"), wordWrap=True)
        hint.setStyleSheet("color: gray")

        self.form = form = QFormLayout(self)
        form.addRow(t("ed_name"), self.name)
        form.addRow(t("ed_gesture"), self.gesture)
        form.addRow(t("ed_sound"), self.sound_row)
        form.addRow(t("ed_mode"), self.sound_mode)
        form.addRow(t("ed_volume"), self.volume_row)
        form.addRow(t("ed_visual"), self.visual_type)
        self.file_label = QLabel()   # says which file, per visual type
        form.addRow(self.file_label, self.visual_file)
        form.addRow(t("ed_text"), self.text)
        form.addRow(t("ed_color"), self.color_btn)
        form.addRow(t("ed_position"), self.position)
        form.addRow(t("ed_size"), self.visual_size)
        form.addRow(t("ed_duration"), self.duration_row)
        for key, field in self.orb_files.items():
            form.addRow(t("orb_" + key), field)
        form.addRow(t("ed_hold"), self.hold)
        form.addRow(self.exclusive)
        form.addRow(hint)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText(t("btn_ok"))
        box.button(QDialogButtonBox.Cancel).setText(t("btn_cancel"))
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

        # show what the jurus holds; the type goes first, as it decides what kind of file the picker takes
        self.visual_type.setCurrentIndex(max(self.visual_type.findData(visual.get("type")), 0))
        self._retype()
        self._select_sound(sound["file"] if isinstance(sound.get("file"), str) else "")
        self.sound_mode.setCurrentIndex(max(self.sound_mode.findData(sound.get("mode")), 0))
        self.volume.setValue(round(_number(sound.get("volume"), 1.0) * 100))
        if isinstance(visual.get("file"), str):
            self.visual_file.set_ref(visual["file"])
        self.text.setText(str(visual.get("text", "")))
        self._set_color(str(visual.get("color", self.color)))
        i = self.position.findData(visual.get("position"))
        self.position.setCurrentIndex(i if i >= 0 else self.position.findData("center"))
        self.visual_size.setValue(_number(visual.get("size"), 0.4))
        duration = visual.get("duration", "held")
        self.held.setChecked(not isinstance(duration, (int, float)))
        self.seconds.setValue(_number(duration, 2.0))
        for key, field in self.orb_files.items():
            if isinstance(orb_sounds.get(key), str):
                field.set_ref(orb_sounds[key])
        self.hold.setValue(_number(j.get("hold"), 0.3))
        self.exclusive.setChecked(bool(j.get("exclusive")))

        self.sound_src.currentIndexChanged.connect(self._sound_changed)
        self.sound_mode.currentIndexChanged.connect(self._sync)
        self.volume.valueChanged.connect(self._sync)
        self.visual_type.currentIndexChanged.connect(self._retype)
        self.held.toggled.connect(self._sync)
        self.color_btn.clicked.connect(self._pick_color)
        self.preview_btn.clicked.connect(self._preview)
        self._sync()

    @staticmethod
    def _row(*widgets) -> QWidget:
        """The widgets side by side as one form row."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            layout.addWidget(w, 1 if isinstance(w, (QComboBox, QSlider)) else 0)
        return row

    # --- the fields that depend on each other ---------------------------------------
    def _retype(self):
        """A new visual type: its file picker takes another kind of file, and the old file goes."""
        kind = self.visual_type.currentData()
        if kind in FILE_VISUALS:
            empty = {"image": "pick_file", "video": "pick_file", "orb": "device_default", "vanish": "choice_none"}[kind]
            self.visual_file.configure(IMAGE_FILES if kind == "image" else VIDEO_FILES, t(empty))
            self.file_label.setText(t("ed_file_" + kind))
        self._sync()

    def _sync(self):
        """Show the rows that apply to the sound and the visual chosen; a looping sound can't be exclusive."""
        kind, ref = self.visual_type.currentData(), self.sound_src.currentData()
        has_sound = kind != "orb" and bool(ref)   # an orb plays its own sounds
        loop = has_sound and self.sound_mode.currentData() == "loop"
        shaped = kind in SHAPED_VISUALS
        for widget, shown in ((self.sound_row, kind != "orb"), (self.sound_mode, has_sound),
                              (self.volume_row, has_sound), (self.visual_file, kind in FILE_VISUALS),
                              (self.text, kind == "text"), (self.color_btn, kind == "text"),
                              (self.position, shaped), (self.visual_size, shaped), (self.duration_row, shaped),
                              *((f, kind == "orb") for f in self.orb_files.values())):
            self.form.setRowVisible(widget, shown)
        self.preview_btn.setEnabled(has_sound)
        self.volume_label.setText(f"{self.volume.value()}%")
        self.seconds.setEnabled(not self.held.isChecked())
        if loop:
            self.exclusive.setChecked(False)
        self.exclusive.setEnabled(not loop)
        if self.isVisible():
            self.adjustSize()

    def _select_sound(self, ref: str):
        i = self.sound_src.findData(ref)
        if i < 0:   # a file picked now, or saved earlier: listed by its file name, before "File..."
            i = self.sound_src.count() - 1
            self.sound_src.insertItem(i, ref.rsplit("/", 1)[-1], ref)
        self._sound_prev = ref
        self.sound_src.setCurrentIndex(i)

    def _sound_changed(self, i: int):
        if self.sound_src.itemData(i) == CHOOSE_FILE:
            ref = pick_file(self, self.store, SOUND_FILES)
            self.sound_src.blockSignals(True)   # going back to a real entry must not ask again
            self._select_sound(ref or self._sound_prev)
            self.sound_src.blockSignals(False)
        else:
            self._sound_prev = self.sound_src.itemData(i)
        self._sync()

    def _set_color(self, color: str):
        if QColor(color).isValid():
            self.color = QColor(color).name()   # "#rrggbb"
        ink = "black" if QColor(self.color).lightness() > 128 else "white"
        self.color_btn.setStyleSheet(f"background: {self.color}; color: {ink}")
        self.color_btn.setText(self.color)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self.color), self, t("ed_color"))
        if color.isValid():   # not cancelled
            self._set_color(color.name())

    def _preview(self):
        """Play the chosen sound once, at the chosen volume. A problem is logged: a preview must never crash."""
        ref, volume = self.sound_src.currentData(), self.volume.value() / 100
        try:
            if ref.startswith("builtin:"):
                # on the monitor mixer only: AudioEngine.play would feed the meeting mixer too, and the others
                # in the meeting would hear the preview
                if sounds := getattr(self.engine, "audio", None):   # the engine opens it when it starts
                    sounds.mixer.play(ref[len("builtin:"):], volume)
            elif path := self.store.resolve(ref):
                if self._player is None:
                    self._out = QAudioOutput(self)
                    self._player = QMediaPlayer(self)
                    self._player.setAudioOutput(self._out)
                self._out.setVolume(volume)
                self._player.setSource(QUrl.fromLocalFile(str(path)))
                self._player.play()
        except Exception:
            log.exception("Could not play the preview of %s", ref)

    # --- the result ---------------------------------------------------------------------
    def value(self) -> dict:
        """The jurus as jurus.yaml holds it: no `sound`, `visual` or `orb_sounds` key when there is none."""
        out = {"name": self.name.text().strip(), "enabled": self._enabled, "gesture": self.gesture.currentData(),
               "hold": self.hold.value(), "exclusive": self.exclusive.isChecked()}
        kind, ref = self.visual_type.currentData(), self.sound_src.currentData()
        if ref and kind != "orb":
            out["sound"] = {"file": ref, "mode": self.sound_mode.currentData(), "volume": self.volume.value() / 100}
        if kind != "none":
            visual = {"type": kind}
            if kind == "text":
                visual.update(text=self.text.text(), color=self.color)
            elif self.visual_file.ref:
                visual["file"] = self.visual_file.ref
            if kind in SHAPED_VISUALS:
                visual.update(position=self.position.currentData(), size=self.visual_size.value(),
                              duration="held" if self.held.isChecked() else self.seconds.value())
            out["visual"] = visual
        chosen = {k: f.ref for k, f in self.orb_files.items() if f.ref}
        if kind == "orb" and chosen:
            out["orb_sounds"] = chosen
        return out

    def _problem(self) -> str | None:
        """What stops the jurus being saved, in words for the user."""
        name, kind = self.name.text().strip(), self.visual_type.currentData()
        if not name:
            return t("err_name_empty")
        if name in self.names:   # the first jurus with a name is the one that fires
            return t("err_name_taken", name=name)
        if kind in ("image", "video") and not self.visual_file.ref:
            return t("err_need_file")
        if kind == "text" and not self.text.text().strip():
            return t("err_need_text")
        return None

    def accept(self):
        if problem := self._problem():
            QMessageBox.warning(self, "MeetCam", problem)   # the dialog stays open
        else:
            super().accept()

    def done(self, result):
        if self._player is not None:
            self._player.stop()
        super().done(result)


class JurusTab(QWidget):
    """The jurus list with Add / Edit / Delete / Try. Every change is saved, rebuilt and handed to the engine at once."""

    def __init__(self, store, engine):
        super().__init__()
        self.store, self.engine = store, engine
        self.sr = engine.cfg["audio"]["sample_rate"]
        self.items = [_dict(c) for c in store.load_jurus()]   # an entry that is no mapping shows as a problem to delete
        self.list = QListWidget(wordWrap=True)
        self.list.itemChanged.connect(self._toggled)
        self.list.itemDoubleClicked.connect(lambda _: self.edit())
        self.list.currentRowChanged.connect(self._sync_buttons)
        self.add_btn, self.edit_btn, self.delete_btn, self.try_btn = (
            QPushButton(t(k)) for k in ("btn_add", "btn_edit", "btn_delete", "btn_try"))
        for button, action in ((self.add_btn, self.add), (self.edit_btn, self.edit), (self.delete_btn, self.delete),
                               (self.try_btn, self.try_)):
            button.clicked.connect(lambda _=False, action=action: action())
        buttons = QHBoxLayout()
        for b in (self.add_btn, self.edit_btn, self.delete_btn, self.try_btn):
            buttons.addWidget(b)
        layout = QVBoxLayout(self)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)
        # ponytail: this builds what the engine has just built, to learn the problems; read them off the engine
        # if start-up gets slow
        self._show(self._build()[0])

    def _build(self):
        templates = valid_templates(self.store.load_gestures())
        return (*build(self.items, self.store, templates, self.sr), templates)

    def refresh(self):
        """The gestures changed, and a rename may have rewritten jurus.yaml: read the list again, then apply it."""
        self.items = [_dict(c) for c in self.store.load_jurus()]
        self.apply()

    def apply(self, show: bool = True):
        """Save the list, rebuild the jurus from it and hand them to the engine; then redraw the list."""
        self.store.save_jurus(self.items)
        # ponytail: media loads in the UI thread on save (a long clip can freeze the window ~1 s); move to a worker
        # if it bites
        built = self._build()
        self.engine.reload(*built)
        if show:
            self._show(built[0])

    # --- the list -----------------------------------------------------------------------
    def _summary(self, cfg: dict) -> str:
        sound, kind = _dict(cfg.get("sound")).get("file"), _dict(cfg.get("visual")).get("type")
        parts = [gesture_label(cfg.get("gesture"))]
        if kind != "orb":   # an orb plays its own sounds
            parts.append(str(sound).removeprefix("builtin:").rsplit("/", 1)[-1] if sound else t("sum_no_sound"))
        parts.append(t("v_" + kind) if kind in VISUALS[1:] else t("sum_no_visual"))
        return f"{cfg.get('name', '')} — " + " · ".join(parts)

    def _show(self, built):
        """Redraw the list; `built` (the Jurus from rules.build, in the same order) tells which ones have a problem."""
        row = max(self.list.currentRow(), 0)   # something is selected, unless the list is empty
        self.list.blockSignals(True)   # filling the list is not the user ticking boxes
        self.list.clear()
        for cfg, j in zip(self.items, built):
            item = QListWidgetItem(("⚠ " if j.problem else "") + self._summary(cfg))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if cfg.get("enabled", True) else Qt.Unchecked)
            item.setToolTip(t("problem_" + j.problem) if j.problem else "")
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.list.setCurrentRow(min(row, self.list.count() - 1))
        self._sync_buttons()

    def _sync_buttons(self):
        """Edit, Delete and Try need a selected jurus."""
        for button in (self.edit_btn, self.delete_btn, self.try_btn):
            button.setEnabled(self.list.currentRow() >= 0)

    def _toggled(self, item):
        cfg, on = self.items[self.list.row(item)], item.checkState() == Qt.Checked
        if on != cfg.get("enabled", True):
            cfg["enabled"] = on
            self.apply(show=False)   # a problem doesn't depend on being enabled, so there is nothing to redraw

    # --- the buttons --------------------------------------------------------------------
    def _selected(self) -> int | None:
        row = self.list.currentRow()
        return row if row >= 0 else None

    def _edit(self, index: int | None):
        others = [c for i, c in enumerate(self.items) if i != index]
        taken = {_dict(c.get("visual")).get("type") for c in others} & {"orb", "vanish"}
        names = {str(c.get("name", "")) for c in others}
        editor = JurusEditor(self.store, None if index is None else self.items[index], taken, engine=self.engine,
                             names=names, parent=self)
        value = editor.value() if editor.exec() == QDialog.Accepted else None
        editor.deleteLater()
        if value is None:
            return
        if index is None:
            self.items.append(value)
        else:
            self.items[index] = value
        self.apply()
        self.list.setCurrentRow(len(self.items) - 1 if index is None else index)

    def add(self):
        self._edit(None)

    def edit(self):
        if (row := self._selected()) is not None:
            self._edit(row)

    def delete(self):
        if (row := self._selected()) is None:
            return
        if _confirm_delete(self, t("confirm_delete", name=self.items[row].get("name", ""))):
            del self.items[row]
            self.apply()

    def try_(self):
        if (row := self._selected()) is not None:
            self.engine.try_jurus(str(self.items[row].get("name", "")))


SENSITIVITY_STEPS = (8, 30)   # the sensitivity slider, in tenths: 0.8 to 3.0
SAVE_DELAY_MS = 400           # a slider drag saves once it pauses: saving rebuilds the jurus, which re-decodes media


class GestureTab(QWidget):
    """The built-in gestures and the ones the user recorded: record, re-record, rename, delete, tune the
    sensitivity and watch the live match score. Every change is saved at once, and `gestures_changed` tells the
    Jurus tab to rebuild. MainWindow feeds it the engine's `status` (on_status) and `recorded` (on_recorded)."""
    gestures_changed = Signal()

    def __init__(self, store, engine):
        super().__init__()
        self.store, self.engine = store, engine
        self.items = store.load_gestures()   # all of it: a hand-edited entry we can't use survives a save
        self.pending_name = None             # the name of the recording being made; None when idle
        self._timer = QTimer(self, singleShot=True, interval=SAVE_DELAY_MS)
        self._timer.timeout.connect(self._save)

        self.builtin_list = QListWidget()
        self.builtin_list.setSelectionMode(QAbstractItemView.NoSelection)   # shown, not editable
        self.builtin_list.setFocusPolicy(Qt.NoFocus)
        self.builtin_list.addItems([t("g_" + g) for g in BUILTIN_GESTURES])
        self.builtin_list.setFixedHeight(self.builtin_list.sizeHintForRow(0) * self.builtin_list.count()
                                         + 2 * self.builtin_list.frameWidth())
        self.recorded_list = QListWidget()
        self.recorded_list.currentRowChanged.connect(self._on_selection)
        self.record_btn, self.rerecord_btn, self.rename_btn, self.delete_btn = (
            QPushButton(t(k)) for k in ("btn_record", "btn_rerecord", "btn_rename", "btn_delete"))
        row = QHBoxLayout()
        for button, action in ((self.record_btn, self.record), (self.rerecord_btn, self.rerecord),
                               (self.rename_btn, self.rename), (self.delete_btn, self.delete)):
            button.clicked.connect(lambda _=False, action=action: action())
            row.addWidget(button)
        self.sens, self.sens_value = QSlider(Qt.Horizontal), QLabel()
        self.sens.setRange(*SENSITIVITY_STEPS)
        self.sens.valueChanged.connect(self._sens_changed)
        sens_row = QHBoxLayout()
        sens_row.addWidget(self.sens, 1)
        sens_row.addWidget(self.sens_value)
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        hint = QLabel(t("score_hint"), wordWrap=True)
        hint.setStyleSheet("color: gray")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(t("gest_builtin")))
        layout.addWidget(self.builtin_list)
        layout.addWidget(QLabel(t("gest_recorded")))
        layout.addWidget(self.recorded_list, 1)
        layout.addLayout(row)
        layout.addWidget(QLabel(t("gest_sensitivity"), wordWrap=True))
        layout.addLayout(sens_row)
        layout.addWidget(self.score_bar)
        layout.addWidget(hint)
        self._show()

    # --- the list -----------------------------------------------------------------------
    def _find(self, name) -> dict | None:
        """The usable recording called `name`: the very dict in `items`, so changing it changes what is saved."""
        return next((g for g in valid_templates(self.items) if g["name"] == name), None)

    def _selected(self) -> str | None:
        item = self.recorded_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _warn_save(self, error: OSError):
        QMessageBox.warning(self, "MeetCam", t("err_gest_save", error=error))

    def _write(self, items: list | None = None) -> bool:
        """Save the recordings (any pending slider change with them), or `items`, which are then the recordings. If
        the disk refuses, say so and keep what was held: False."""
        self._timer.stop()
        try:
            self.store.save_gestures(self.items if items is None else items)
        except OSError as e:
            self._warn_save(e)
            return False
        if items is not None:
            self.items = items
        return True

    def _save(self, items: list | None = None) -> bool:
        """_write, then tell the Jurus tab."""
        done = self._write(items)
        if done:
            self.gestures_changed.emit()
        return done

    def _show(self, select: str | None = None):
        """Redraw the recordings with `select` (else what was selected, else the first) selected."""
        select = select or self._selected()
        templates = valid_templates(self.items)
        self.recorded_list.blockSignals(True)   # filling the list is not the user picking a row
        self.recorded_list.clear()
        for g in templates:
            item = QListWidgetItem("🖐 " + g["name"] + (t("gest_hands2") if g["hands"] == 2 else ""))
            item.setData(Qt.UserRole, g["name"])
            self.recorded_list.addItem(item)
        self.recorded_list.blockSignals(False)
        names = [g["name"] for g in templates]
        self.recorded_list.setCurrentRow(names.index(select) if select in names else 0 if names else -1)
        self._on_selection()

    def _on_selection(self, *_):
        """Show the selected recording's sensitivity (without saving it) and clear the score."""
        g = self._find(self._selected())
        if g:
            self.sens.blockSignals(True)   # showing a value is not the user moving the slider
            # a hand-edited file may hold NaN or 1e308; the slider shows the nearest it can
            steps = _number(g["sensitivity"], DEFAULT_SENSITIVITY) * 10
            self.sens.setValue(round(min(max(steps, SENSITIVITY_STEPS[0]), SENSITIVITY_STEPS[1])))
            self.sens.blockSignals(False)
        self.sens.setEnabled(g is not None)
        self._show_sensitivity()
        self._score(None)
        self._sync_buttons()

    def _sync_buttons(self):
        """Re-record, Rename and Delete need a selected recording; nothing starts while the engine is recording."""
        idle, picked = self.pending_name is None, self._selected() is not None
        self.record_btn.setEnabled(idle)
        for button in (self.rerecord_btn, self.rename_btn, self.delete_btn):
            button.setEnabled(idle and picked)

    # --- sensitivity and the live score ----------------------------------------------------
    def _show_sensitivity(self):
        self.sens_value.setText(f"{self.sens.value() / 10:.1f}×" if self.sens.isEnabled() else "")

    def _sens_changed(self, value: int):
        if g := self._find(self._selected()):
            g["sensitivity"] = value / 10
            self._timer.start()   # saved when the slider rests
        self._show_sensitivity()

    def on_status(self, st: dict):
        self._score((st.get("scores") or {}).get(self._selected()))

    def _score(self, score: float | None):
        """The bar for the selected recording's match score: full at 1, empty at 0 or below, "—" when unknown."""
        score = _number(score, None)   # a NaN tolerance in a hand-edited file gives a NaN score
        if score is None:
            self.score_bar.setValue(0)
            self.score_bar.setFormat("—")
        else:
            self.score_bar.setValue(round(max(0.0, min(1.0, score)) * 100))
            self.score_bar.setFormat(t("score_match") if score > 0 else "%p%")

    # --- the buttons --------------------------------------------------------------------
    def _ask_name(self, title: str, current: str = "") -> str | None:
        """What the user typed in a small dialog, or None if cancelled."""
        box = QInputDialog(self)
        box.setWindowTitle(title)
        box.setLabelText(t("gest_name_label"))
        box.setTextValue(current)
        box.setOkButtonText(t("btn_ok"))   # not Qt's English buttons whatever the language
        box.setCancelButtonText(t("btn_cancel"))
        name = box.textValue() if box.exec() == QDialog.Accepted else None
        box.deleteLater()
        return name

    def _new_name(self, title: str, current: str = "") -> str | None:
        """Ask for a name, unless it can't be used: then say why and give None. `current` may be kept."""
        name = self._ask_name(title, current)
        if name is None:
            return None
        name = name.strip()
        problem = (t("err_gest_empty") if not name else t("err_gest_colon") if ":" in name   # "rec:" + name
                   else t("err_gest_taken", name=name) if name != current and self._find(name) else None)
        if problem:
            QMessageBox.warning(self, "MeetCam", problem)
            return None
        return name

    def _start(self, name: str):
        self.pending_name = name
        self._sync_buttons()
        self.engine.record()   # the answer comes to on_recorded

    def record(self):
        if (name := self._new_name(t("gest_name_title"))) is not None:
            self._start(name)

    def rerecord(self):
        if (name := self._selected()) is not None:
            self._start(name)

    def on_recorded(self, result):
        """The engine's answer: the template of the hand(s) it saw, or "no_hand" / "unsteady"."""
        name, self.pending_name = self.pending_name, None
        self._sync_buttons()   # first: whatever goes wrong below, Record must work again
        if name is None:   # not asked for here
            return
        if isinstance(result, dict):
            old = self._find(name)
            template = {**result, "name": name}
            if old:   # a re-recording keeps the tuning
                template["sensitivity"] = _number(old["sensitivity"], template["sensitivity"])
            if self._save([template if g is old else g for g in self.items] if old else [*self.items, template]):
                self._show(select=name)
        else:
            QMessageBox.warning(self, t("record_failed_title"), t("record_" + str(result)))

    def rename(self):
        old = self._selected()
        if old is None:
            return
        new = self._new_name(t("gest_rename_title"), old)
        if new is None or new == old:
            return
        g = self._find(old)
        if not self._write([{**g, "name": new} if x is g else x for x in self.items]):
            return   # nothing changed
        jurus = self.store.load_jurus()
        for j in jurus:   # the jurus that use it follow it
            if isinstance(j, dict) and j.get("gesture") == "rec:" + old:
                j["gesture"] = "rec:" + new
        try:
            self.store.save_jurus(jurus)
        except OSError as e:   # they keep the old name, which the Jurus tab marks with a warning sign
            self._warn_save(e)
        self.gestures_changed.emit()   # after the jurus are written: the Jurus tab reads them again
        self._show(select=new)

    def delete(self):
        name = self._selected()
        if name is None:
            return
        if _confirm_delete(self, t("confirm_delete_gesture", name=name)):
            gone = self._find(name)
            # the jurus that use it stay, marked ⚠ by the Jurus tab's rebuild
            if self._save([g for g in self.items if g is not gone]):
                self._show()


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

        # two short rows of lights and two of buttons, not one long row each: the window's smallest width must
        # fit a 1366 px laptop in Indonesian too
        lights, info = QHBoxLayout(), QHBoxLayout()
        lights.addWidget(self.vcam_light)
        lights.addWidget(self.meeting_light)
        lights.addStretch()
        info.addWidget(self.fps_label)
        info.addWidget(self.gesture_label)
        info.addStretch()
        info.addWidget(self.invisible_badge)

        self.effects = self._button("btn_effects", "Space", lambda on: setattr(self.engine, "effects_on", on), True)
        self.mute = self._button("btn_mute", "M", lambda on: setattr(self.engine, "muted", on))
        self.skeleton = self._button("btn_skeleton", "D", self._set_skeleton, bool(settings["skeleton"]))
        self.room = QPushButton(t("btn_room"))
        self.room.clicked.connect(self.engine.capture_room)
        QShortcut(QKeySequence("B"), self, activated=self.room.click)
        QShortcut(QKeySequence("Q"), self, activated=self.close)
        buttons = QGridLayout()
        for i, b in enumerate((self.effects, self.mute, self.room, self.skeleton)):
            b.setFocusPolicy(Qt.NoFocus)  # so Space is the Effects shortcut, never "press the focused button"
            buttons.addWidget(b, i // 2, i % 2)

        self.tabs = QTabWidget()
        self.jurus_tab = JurusTab(store, self.engine)   # the first tab, so the one shown
        self.gesture_tab = GestureTab(store, self.engine)
        self.settings_tab = SettingsTab(store, self.engine, cfg)
        self.tabs.addTab(self.jurus_tab, t("tab_jurus"))
        self.tabs.addTab(self.gesture_tab, t("tab_gestures"))
        self.tabs.addTab(self.settings_tab, t("tab_settings"))
        self.gesture_tab.gestures_changed.connect(self.jurus_tab.refresh)
        self.engine.status.connect(self.gesture_tab.on_status)
        self.engine.recorded.connect(self.gesture_tab.on_recorded)

        root = QWidget()
        left = QVBoxLayout()
        left.addWidget(self.preview, 1)
        left.addWidget(self.message)
        left.addLayout(lights)
        left.addLayout(info)
        left.addLayout(buttons)
        columns = QHBoxLayout(root)
        columns.addLayout(left, 1)  # the preview takes the width the tabs don't need
        columns.addWidget(self.tabs)
        self.setCentralWidget(root)
        self.resize(1200, 680)  # fits a 1366x768 laptop screen

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
