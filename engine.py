# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 lfarabi
"""The frame loop, in a QThread: camera -> hand tracking -> jurus (rules, orb, vanish) -> virtual camera,
plus a preview image and a status dict for the window.

The UI calls the public methods from its own thread. Commands are queued and run here, between frames;
`reload` builds nothing on this thread: the loop swaps in what the UI built before its next frame."""
import logging
import math
import queue
import sys
import threading
import time
from typing import NamedTuple

import cv2
import numpy as np
import pyvirtualcam
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

import poses
from audio import AudioEngine
from gestures import HIDDEN, NO_HAND, GestureClassifier, OrbMachine
from jutsu import Vanish
from orb import EnergyOrb, occlude_by_hand, orb_anchor
from rules import Jurus, Rules, active_ids, build, hands_center, valid_templates
from tracking import CONNECTIONS, WRIST, HandTracker, Smoother, palm_center, palm_normal, palm_size, pick_primary

log = logging.getLogger(__name__)
RECORD_COUNTDOWN, RECORD_SECONDS = 3.0, 2.0  # recording a gesture: time to strike the pose, then time spent collecting it
LOST_AFTER = 30                              # this many failed reads in a row and the camera counts as gone


def open_camera(cfg: dict) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg["device_id"], cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    # MJPG must be set before the size, otherwise most webcams fall back to slow raw YUY2 at 720p.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
    return cap


def open_virtual_camera(cfg: dict, width: int, height: int, fps: int) -> pyvirtualcam.Camera | None:
    """The output Meet/Zoom pick as a camera. Missing or busy driver → preview-only, not a crash."""
    if not cfg["enabled"]:
        return None
    try:
        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, backend=cfg["backend"],
                                  fmt=pyvirtualcam.PixelFormat.BGR)
    except Exception as e:
        log.warning("Virtual camera unavailable, running preview-only: %s "
                    "(is OBS Studio's own Start Virtual Camera active? Stop it or close OBS)", e)
        return None
    log.info("Virtual camera: sending %dx%d to \"%s\"; select it as the camera in Meet/Zoom", width, height, cam.device)
    return cam


def draw_skeleton(img, lm):
    pts = lm[:, :2].astype(int)
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 4, (0, 200, 255), -1, cv2.LINE_AA)


def _visual_file(store, j: Jurus) -> str | None:
    """Disk path of the media file a jurus's visual points at, if it has one."""
    ref = j.cfg["visual"].get("file")
    path = store.resolve(ref) if isinstance(ref, str) else None
    return str(path) if path else None


class _Setup(NamedTuple):
    """What the loop runs on, built by `reload` and swapped in whole between two frames."""
    rules: Rules
    orb_jurus: Jurus | None
    machine: OrbMachine | None   # the orb's state machine and renderer exist only with an orb jurus
    orb: EnergyOrb | None
    templates: list[dict]
    smoke: str | None            # the vanish jurus's smoke clip; the Vanish itself needs the frame size
    sounds: dict
    loops: set


class Engine(QThread):
    frame_ready = Signal(QImage)  # the preview, BGR; the next one waits for frame_shown()
    # keys: fps, vcam, meeting, gesture, message (a key, "" for none), message_args, scores, invisible, effects_on, muted
    status = Signal(dict)
    recorded = Signal(object)     # a recorded gesture's template (dict), or "no_hand" / "unsteady"

    def __init__(self, cfg: dict, store, settings: dict, tracker=None, audio=None, capture=None):
        """Whatever is left None is opened when the thread starts."""
        super().__init__()
        self.cfg, self.store, self.settings = cfg, store, dict(settings)
        self.tracker, self.audio, self.cap = tracker, audio, capture
        self.vcam = None
        self.vanish = None   # made once the frame size is known
        self.effects_on, self.skeleton = True, bool(settings["skeleton"])
        self._muted = False
        self._cmds = queue.SimpleQueue()
        self._running = True
        self._fps, self._last, self._drops, self._oops_at = 0.0, time.perf_counter(), 0, float("-inf")
        self._lock, self._pending, self._setup = threading.Lock(), None, None
        self._preview_free = True  # the window has taken the last preview frame
        self._shape, self._vanish_key = None, None
        self._rec = None   # a recording in progress: {"start": time or None, "frames": [hands, ...]}
        self._info = {"gesture": "", "message": "", "message_args": {}, "scores": {}, "invisible": False}
        t = cfg["tracking"]
        self._smoother, self._world_smoother = Smoother(t["smoothing"]), Smoother(cfg["orb"]["world_smoothing"])
        self._classifier = GestureClassifier(cfg["gesture"])
        self._normal, self._palm_side, self._anchor = None, 1.0, None  # palm direction; where the orb last was
        self._two_hands_at = float("-inf")  # last time two hands were seen
        self._audio_state = HIDDEN          # state last reported to audio; forced to HIDDEN while effects are off
        templates = store.load_gestures()
        self.reload(*build(store.load_jurus(), store, templates, cfg["audio"]["sample_rate"]), templates)
        self._apply_pending()

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool):
        self._muted = value  # SFX only, for you and the meeting; your voice keeps going
        if self.audio:
            self.audio.muted = value

    # --- called from the UI thread ------------------------------------------------
    def reload(self, jurus: list[Jurus], sounds: dict, loops: set, templates: list[dict]):
        """Use these jurus (from rules.build) and recorded gestures. Builds the orb here, in the caller's
        thread; the loop switches over at the start of its next frame."""
        rules = Rules(jurus, self.audio)
        orb_j = rules.orb_jurus
        machine = orb = None
        if orb_j:
            video = _visual_file(self.store, orb_j)
            with self._lock:
                last = self._pending or self._setup
            if last and last.machine and (last.machine.trigger, last.orb.o["video"]) == (orb_j.gesture, video):
                machine, orb = last.machine, last.orb   # unchanged: the orb doesn't blink or replay its sound
            else:
                machine = OrbMachine(self.cfg, trigger=orb_j.gesture)
                orb = EnergyOrb({**self.cfg, "orb": {**self.cfg["orb"], "style": "video" if video else "procedural",
                                                     "video": video}})
        vanish_j = next((j for j in jurus if j.kind == "vanish" and j.problem is None), None)
        smoke = _visual_file(self.store, vanish_j) if vanish_j else None
        with self._lock:
            self._pending = _Setup(rules, orb_j, machine, orb, valid_templates(templates), smoke, sounds, loops)

    def frame_shown(self):
        """The window has taken the last preview frame: the next one may be sent."""
        self._preview_free = True

    def _send(self, name: str, *args):
        self._cmds.put((name, args))

    def capture_room(self):
        self._send("capture_room")

    def record(self):
        self._send("record")

    def try_jurus(self, name: str):
        self._send("try_jurus", name)

    def set_settings(self, settings: dict):
        self._send("set_settings", dict(settings))

    def stop(self):
        self._send("stop")

    # --- the thread -----------------------------------------------------------------
    def run(self):
        try:
            try:
                self._start()
            except Exception:
                self._oops("Starting")  # carry on: the loop idles on whatever did open, and still obeys stop()
            self._last = time.perf_counter()
            while self._running:
                try:
                    self._iterate()
                except Exception:  # one bad frame or file must not end the loop
                    self._oops("The frame loop")
                    time.sleep(0.01)  # a failure that repeats every pass must not spin
        finally:
            self._close()

    def _oops(self, what: str):
        """Log a failure the loop carries on after; at most every 5 s, as it may repeat every frame."""
        now = time.monotonic()
        if now - self._oops_at > 5:
            self._oops_at = now
            log.exception("%s failed; carrying on (repeats are not logged for 5 s)", what)

    def _iterate(self):
        self._drain()
        if self.cap is None or self.tracker is None:  # nothing to run on: wait for new settings, or for stop
            time.sleep(0.2)
            return
        ok, frame = self.cap.read()
        if not ok:
            self._drops += 1
            if self._drops >= LOST_AFTER:  # unplugged or taken by another app: stop spinning, say so
                self._drops = 0
                self._release_camera()
                self._emit_status(message="camera_failed")
            time.sleep(0.01)
            return
        self._drops = 0
        now = time.perf_counter()
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._last, 1e-6)
        self._last = now
        raw = frame.copy()  # step draws into `frame`
        try:
            outgoing, preview = self.step(frame, now)
            status = self._status(self._fps)
        except Exception:  # e.g. a bad file: the meeting gets the plain camera, not a frozen picture
            self._oops("Drawing the effects")
            outgoing, preview = raw, cv2.flip(raw, 1) if self.settings["mirror_preview"] else raw
            status = {**self._status(self._fps), "message": "engine_error", "message_args": {}}
        if self.vcam is not None and outgoing.shape[:2] == (self.vcam.height, self.vcam.width):
            self.vcam.send(outgoing)
        if self._preview_free:  # else the window is busy, and frames queued for it would pile up
            self._preview_free = False
            h, w = preview.shape[:2]
            self.frame_ready.emit(QImage(preview.data, w, h, preview.strides[0], QImage.Format.Format_BGR888).copy())
        self.status.emit(status)

    def _close(self):
        for close in (self.vcam and self.vcam.close, self.audio and self.audio.close,
                      self.tracker and self.tracker.close, self._release_camera):
            try:
                if close:
                    close()
            except Exception:  # the others must still be closed, above all the camera
                log.exception("Closing a device failed")

    def _start(self):
        if self.audio is None:
            self._open_audio()
        if self.tracker is None:
            try:
                self.tracker = HandTracker(self.cfg["tracking"])
            except Exception as e:  # e.g. the model file isn't downloaded (it is not in git)
                log.error("Hand tracker failed to start: %s", e)
                self._emit_status(message="tracker_failed")
                return
        self._open_camera(self.cap)
        self._open_vcam()

    def _drain(self):
        try:
            self._apply_pending()  # so a command queued after a reload sees the new jurus
        except Exception:
            self._oops("Switching to the new jurus")  # stop() must still get through
        while True:
            try:
                name, args = self._cmds.get_nowait()
            except queue.Empty:
                return
            try:
                getattr(self, "_do_" + name)(*args)
            except Exception:
                log.exception("Command %s failed", name)

    # --- one frame --------------------------------------------------------------------
    def step(self, frame: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Run one camera frame through everything. Returns (outgoing, preview): `outgoing` is `frame` with
        the effects drawn in, unmirrored, for the meeting; `preview` is the copy for the window."""
        self._shape = frame.shape
        self._apply_pending()
        s, vanish = self._setup, self.vanish
        hands, hand, lm, gesture, two_hands = self._track(frame, t)
        active = active_ids(gesture, hands, hand, lm, frame.shape[0], self.cfg, s.templates)
        state, occlusion = self._update_orb(active, gesture, two_hands, lm, t)

        vanish.capture_tick(frame, t)  # raw camera frame, before any effect
        if self.effects_on:
            # the pose being recorded must not fire into the meeting
            busy = bool(s.machine and s.machine.busy(t)) or vanish.busy or self._rec is not None
            for j in s.rules.update(active, t, two_hands, busy):
                if j.kind == "vanish":
                    self._fire_vanish(hands, t)
            vanish.update(t)
            vanish.apply_plate(frame)
            if s.machine and self._anchor is not None:
                # an invisible hand doesn't cover the orb
                original = frame.copy() if occlusion > 0.01 and state != HIDDEN and vanish.mix < 0.5 else None
                s.orb.draw(frame, s.machine, *self._anchor, t)
                if original is not None:
                    occlude_by_hand(frame, original, lm, occlusion)
            vanish.draw_smoke(frame, t)
        else:
            s.rules.update(set(), t, False, False)  # releases whatever is held
            state = HIDDEN
        m = s.machine
        self.audio.update(self._audio_state, state, *((m.charge, m.blast_power, m.silent_fade) if m else (0.0, 0.0, False)))
        self._audio_state = state
        preview = self._render(frame, lm, t)

        message, args = self._record_tick(hands, t) if self._rec else ("", {})
        if not message:
            message = vanish.message(t)
            args = {"seconds": vanish.countdown(t)} if message == "room_countdown" else {}
        self._info = {"gesture": ", ".join(sorted(active)), "message": message, "message_args": args,
                      "scores": {tp["name"]: poses.score(tp, hands, hand) for tp in s.templates},
                      "invisible": vanish.mix >= 0.5}
        return frame, preview

    def _track(self, frame, t: float):
        """Hands in this frame: (all hands, the one that drives the orb, its smoothed landmarks, its
        gesture, whether two hands are up)."""
        hands = self.tracker.detect(frame, int(t * 1000))
        hand = pick_primary(hands, None if self._smoother.value is None else self._smoother.value[WRIST, :2])
        lm = self._smoother.update(hand.lm if hand else None)
        world = self._world_smoother.update(hand.world if hand else None)
        gesture = self._classifier.update(lm, world)
        if lm is not None:
            self._normal, self._palm_side = palm_normal(world, self._palm_side)
        # Two hands up = a two-hand pose forming or held. While crossed or touching, MediaPipe often loses one
        # hand for a few frames, so "two hands" lingers for a grace period: the lone hand seen then is half
        # of a pose, not a finger count, fist or orb.
        if len(hands) > 1:
            self._two_hands_at = t
        return hands, hand, lm, gesture, t - self._two_hands_at < self.cfg["gesture"]["two_hand_grace"]

    def _update_orb(self, active: set, gesture: str, two_hands: bool, lm, t: float) -> tuple[str, float]:
        """Give the orb its gesture. Returns its state and how much the hand covers it (0..1)."""
        s = self._setup
        if not s.machine:
            return HIDDEN, 0.0
        j = s.orb_jurus
        # a recorded match outranks the built-in gesture it replaced; any other raw gesture still goes in, as the
        # FOUR tolerance and the silent fade need it
        orb_in = (NO_HAND if self.vanish.busy or self._rec is not None or (two_hands and j.hands == 1)
                  else j.gesture if j.gesture in active else NO_HAND if gesture == j.gesture else gesture)
        state = s.machine.update(orb_in, -float(self._normal[2]) if lm is not None else None, t)  # palm-to-camera
        if lm is None:
            return state, 0.0
        o = self.cfg["orb"]
        self._anchor = orb_anchor(lm, self._normal, o)
        return state, float(np.clip((self._normal[2] - o["occlusion_start"]) / (o["occlusion_full"] - o["occlusion_start"]), 0, 1))

    def _render(self, frame, lm, t: float) -> np.ndarray:
        """Draw the overlays into `frame` (for the meeting) and return the preview. They go on after the
        mirroring, so text reads the right way round in both; positions are as seen in the preview."""
        rules, mirrored = self._setup.rules, self.settings["mirror_preview"]
        palm = None if lm is None else (*(float(x) for x in palm_center(lm)[:2]), palm_size(lm))
        preview = frame.copy()
        if self.skeleton and lm is not None:
            draw_skeleton(preview, lm)
        if mirrored:
            preview = cv2.flip(preview, 1)
        rules.draw(frame, t, palm, False, mirrored)
        rules.draw(preview, t, palm, True, mirrored)
        return preview

    def _fire_vanish(self, hands, t: float):
        at = hands_center(hands)
        h, w = self._shape[:2]
        self.vanish.fire((w / 2, h / 2) if at is None else at, t)

    def _record_tick(self, hands, t: float) -> tuple[str, dict]:
        """Advance the recording; returns the status message to show."""
        r = self._rec
        if r["start"] is None:
            r["start"] = t
        since = t - r["start"]
        if since < RECORD_COUNTDOWN:
            return "record_countdown", {"seconds": math.ceil(RECORD_COUNTDOWN - since)}
        if since < RECORD_COUNTDOWN + RECORD_SECONDS:
            r["frames"].append(hands)
            return "recording", {}
        self._rec = None
        try:
            result = poses.make_template("", r["frames"])
        except ValueError as e:  # "no_hand" | "unsteady"
            result = str(e)
        self.recorded.emit(result)
        return "", {}

    def _status(self, fps: float) -> dict:
        return {**self._info, "fps": fps, "vcam": self.vcam is not None,
                "meeting": bool(self.audio and self.audio.meeting_on), "effects_on": self.effects_on, "muted": self.muted}

    def _emit_status(self, fps: float = 0.0, **over):
        self.status.emit({**self._status(fps), **over})

    # --- swapping in a reload ------------------------------------------------------------
    def _apply_pending(self):
        with self._lock:  # in one step, so reload() always compares the orb with the newest setup
            old, setup, self._pending = self._setup, self._pending, None
            if setup is not None:
                self._setup = setup
        if setup is not None:
            if old is not None:
                old.rules.release_all(time.perf_counter())
            self._attach_audio()
        if self._shape is not None:
            key = (self._setup.smoke, self._shape)
            if key != self._vanish_key:  # unchanged smoke and size: keep it, so being invisible survives an edit
                self.vanish = Vanish({"smoke": key[0], "background": str(self.store.background),
                                      **self.cfg["vanish_defaults"]}, self._shape)
                self._vanish_key = key

    def _attach_audio(self):
        if self.audio:
            self.audio.set_sounds(self._setup.sounds, self._setup.loops)
            self._setup.rules.audio = self.audio

    # --- devices -------------------------------------------------------------------------
    def _open_camera(self, cap=None) -> bool:
        """Open the camera chosen in the settings (or take `cap`) and read one frame to learn its size.
        If it gives none, say so and return False."""
        try:
            if cap is None:
                self._release_camera()
                cap = open_camera({**self.cfg["camera"], "device_id": self.settings["camera_index"]})
            ok, frame = cap.read()
        except Exception as e:  # e.g. a camera_index in settings.yaml that is not a number
            log.warning("Camera could not be opened: %s", e)
            ok = False
        if not ok:
            if cap is not None:
                cap.release()
            self.cap = None
            self._emit_status(message="camera_failed")
            return False
        self.cap, self._shape = cap, frame.shape
        return True

    def _release_camera(self):
        cap, self.cap = self.cap, None
        if self._rec:  # a recording can't finish without frames
            self._rec = None
            self.recorded.emit("no_hand")
        if cap:
            cap.release()

    def _open_audio(self):
        if self.audio:
            self.audio.close()
        s = self.settings
        try:
            self.audio = AudioEngine({**self.cfg["audio"], "output_device": s["speaker"]},
                                     {**self.cfg["meeting_audio"], "mic_device": s["mic"], "output_device": s["meeting_output"]})
        except Exception:  # e.g. no PortAudio: run silent rather than not at all
            log.exception("Audio unavailable, running silent")
            self.audio = AudioEngine({**self.cfg["audio"], "enabled": False})
        self.audio.muted = self._muted
        self._attach_audio()

    def _open_vcam(self):
        if self.vcam:
            self.vcam.close()
        self.vcam = None
        if self.cap is not None:
            h, w = self._shape[:2]
            cfg = {**self.cfg["virtual_camera"], "enabled": self.settings["virtual_camera"]}
            self.vcam = open_virtual_camera(cfg, w, h, self.cfg["camera"]["fps"])

    # --- commands, run between frames ------------------------------------------------------
    def _do_capture_room(self):
        if self.vanish:
            self.vanish.start_capture(time.perf_counter())

    def _do_record(self):
        if self.cap is None or self.tracker is None:  # no frames are coming
            self.recorded.emit("no_hand")
        else:
            self._rec = {"start": None, "frames": []}

    def _do_try_jurus(self, name: str):
        now = time.perf_counter()
        j = self._setup.rules.try_fire(name, now)
        if j and j.kind == "vanish" and self.vanish:
            self._fire_vanish([], now)

    def _do_set_settings(self, new: dict):
        old, self.settings = self.settings, new
        camera = self.cap is None or new["camera_index"] != old["camera_index"]  # also retries a failed camera
        if camera:
            self._open_camera()
        if any(new[k] != old[k] for k in ("mic", "speaker", "meeting_output")):
            self._open_audio()
        if camera or new["virtual_camera"] != old["virtual_camera"]:
            self._open_vcam()

    def _do_stop(self):
        self._running = False
