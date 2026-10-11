import time

import numpy as np
import soundfile as sf
from conftest import FakeTracker, make_engine
from test_logic import HORIZONTAL, VERTICAL, hand3d, pick

import engine
from gestures import ACTIVE, FADING, HIDDEN
from poses import make_template
from rules import build
from store import DEFAULT_SETTINGS, Store
from tracking import Hand

DT = 0.04
BLACK = lambda w=640, h=360: np.zeros((h, w, 3), np.uint8)  # noqa: E731


# --- virtual camera -------------------------------------------------------------
def test_virtual_camera_failure_falls_back_to_preview_only(monkeypatch):
    def busy(**kwargs):
        raise RuntimeError("virtual camera output could not be started")
    monkeypatch.setattr(engine.pyvirtualcam, "Camera", busy)
    assert engine.open_virtual_camera({"enabled": True, "backend": "obs"}, 1280, 720, 30) is None


def test_virtual_camera_can_be_disabled():
    assert engine.open_virtual_camera({"enabled": False, "backend": "obs"}, 1280, 720, 30) is None


# --- a frame ------------------------------------------------------------------------
def test_step_returns_unmirrored_output_and_mirrored_preview(tmp_path, qapp):
    e = make_engine(Store(tmp_path))
    f = BLACK()
    f[:, :10] = 255
    out, prev = e.step(f, 0.0)
    assert out[:, :10].min() == 255 and prev[:, -10:].min() == 255


def test_preview_matches_the_output_when_mirroring_is_off(tmp_path, qapp):
    e = make_engine(Store(tmp_path), {**DEFAULT_SETTINGS, "mirror_preview": False})
    f = BLACK()
    f[:, :10] = 255
    out, prev = e.step(f, 0.0)
    assert np.array_equal(out, prev) and prev[:, :10].min() == 255


def test_text_overlay_reads_the_same_in_preview_and_output(tmp_path, qapp):
    store = Store(tmp_path)
    store.save_jurus([{"name": "WOW", "enabled": True, "gesture": "WOW", "hold": 0.3, "exclusive": False,
                       "visual": {"type": "text", "text": "WOW!", "color": "#ffd400", "position": "top_left",
                                  "size": 0.25, "duration": 2.0}}])
    e = make_engine(store)
    e._do_try_jurus("WOW")
    out, prev = e.step(BLACK(), time.perf_counter())

    def sprite(img):  # the lit pixels' bounding box: (x0, crop)
        ys, xs = np.nonzero(img.any(axis=2))
        return xs.min(), img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    (px, in_preview), (ox, in_output) = sprite(prev), sprite(out)
    assert px < 30 and ox + in_output.shape[1] > 640 - 30   # "top left" in the preview, so the right edge in the meeting
    assert np.array_equal(in_preview, in_output)  # but the letters themselves are not flipped in either


def test_reload_swaps_jurus_between_frames(tmp_path, qapp):
    store = Store(tmp_path)
    t = np.arange(4800) / 48000
    sf.write(store.media / "beep.wav", (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32), 48000)
    e = make_engine(store)
    e.step(BLACK(), 0.0)
    old, vanish = e._setup, e.vanish
    jurus, sounds, loops = build([{"name": "Beep", "enabled": True, "gesture": "ONE", "hold": 0.3,
                                   "exclusive": False, "sound": {"file": "media/beep.wav", "mode": "once", "volume": 1.0}},
                                  {"name": "Menghilang", "enabled": True, "gesture": "SEAL", "hold": 0.3,
                                   "exclusive": True, "visual": {"type": "vanish"}}], store, [], 48000)
    e.reload(jurus, sounds, loops, [])
    assert e._setup is old and "media/beep.wav" not in e.audio.sounds   # nothing changes mid-frame
    e.step(BLACK(), DT)
    assert [j.name for j in e._setup.rules.jurus] == ["Beep", "Menghilang"] and "media/beep.wav" in e.audio.sounds
    assert e._setup.orb_jurus is None and e._setup.rules.audio is e.audio
    assert e.vanish is vanish                                          # same smoke: being invisible survives an edit


# --- the orb ----------------------------------------------------------------------
def orb_engine(tmp_path, gesture, hands, templates=()):
    store = Store(tmp_path)
    store.save_gestures(list(templates))
    store.save_jurus([{"name": "Orb", "enabled": True, "gesture": gesture, "hold": 0.3, "exclusive": False,
                       "visual": {"type": "orb"}}])
    return make_engine(store, tracker=FakeTracker(hands))


def run_frames(e, seconds, t0):
    for i in range(round(seconds / DT)):
        out, _ = e.step(BLACK(1280, 720), t0 + i * DT)
    return out


def test_the_orb_is_summoned_by_its_jurus_gesture_and_drawn(tmp_path, qapp):
    peace = Hand(*hand3d(pick(2), False))
    e = orb_engine(tmp_path, "TWO", [peace])
    out = run_frames(e, 0.5, time.perf_counter())
    assert e._setup.machine.state == ACTIVE and out.any()
    e.effects_on = False
    assert not run_frames(e, 0.2, time.perf_counter() + 1).any()   # effects off: nothing drawn
    other = orb_engine(tmp_path, "FIST", [peace])                    # a different gesture leaves the orb away
    run_frames(other, 0.5, time.perf_counter())
    assert other._setup.machine.state == HIDDEN


def test_the_orb_can_be_summoned_by_a_recorded_gesture(tmp_path, qapp):
    peace = Hand(*hand3d(pick(2), False))
    e = orb_engine(tmp_path, "rec:peace", [peace], [make_template("peace", [[peace]] * 10)])
    run_frames(e, 0.5, time.perf_counter())
    assert e._setup.machine.state == ACTIVE and e._status(0.0)["gesture"] == "rec:peace"
    assert e._status(0.0)["scores"]["peace"] > 0.9


def test_a_one_hand_orb_gives_way_to_a_two_hand_pose(tmp_path, qapp):
    peace = Hand(*hand3d(pick(2), False))
    e = orb_engine(tmp_path, "TWO", [peace])
    t = time.perf_counter()
    run_frames(e, 0.5, t)
    assert e._setup.machine.state == ACTIVE
    e.tracker.hands = [peace, peace]
    run_frames(e, 0.2, t + 1)
    assert e._setup.machine.state in (FADING, HIDDEN)


# --- the camera -----------------------------------------------------------------
def test_missing_camera_reports_instead_of_crashing(tmp_path):
    e = make_engine(Store(tmp_path))
    e.settings = {**DEFAULT_SETTINGS, "camera_index": 97}
    msgs = []
    e.status.connect(msgs.append)
    assert e._open_camera() is False and msgs[-1]["message"] == "camera_failed"


class FakeCap:
    def __init__(self):
        self.released = False

    def read(self):
        time.sleep(0.005)
        return True, BLACK()

    def release(self):
        self.released = True


class FakeVcam:
    closed = False

    def close(self):
        self.closed = True


def test_changing_settings_reopens_only_what_changed(tmp_path, qapp, monkeypatch):
    e = make_engine(Store(tmp_path))
    first = FakeCap()
    assert e._open_camera(first)
    opened = []
    monkeypatch.setattr(engine, "open_camera", lambda cfg: opened.append(cfg["device_id"]) or FakeCap())
    monkeypatch.setattr(engine, "open_virtual_camera", lambda cfg, w, h, fps: FakeVcam() if cfg["enabled"] else None)
    e._open_vcam()
    vcam, audio = e.vcam, e.audio
    assert isinstance(vcam, FakeVcam)

    e._do_set_settings({**e.settings, "mirror_preview": False, "language": "id"})
    assert not opened and e.audio is audio and e.vcam is vcam and not first.released

    e.muted = True
    e._do_set_settings({**e.settings, "speaker": "Headphones"})
    assert e.audio is not audio and e.audio.muted and e._setup.rules.audio is e.audio and e.vcam is vcam

    e._do_set_settings({**e.settings, "camera_index": 3})
    assert opened == [3] and first.released and e.vcam is not vcam and vcam.closed

    e._do_set_settings({**e.settings, "virtual_camera": False})
    assert e.vcam is None


def test_run_streams_frames_until_stopped(tmp_path, qapp):
    cap = FakeCap()
    e = make_engine(Store(tmp_path), {**DEFAULT_SETTINGS, "virtual_camera": False})
    e.cap = cap
    frames, statuses = [], []
    e.frame_ready.connect(frames.append)
    e.status.connect(statuses.append)
    e.start()
    deadline = time.time() + 10
    while len(frames) < 3 and time.time() < deadline:
        qapp.processEvents()  # the signals are queued to this thread
        time.sleep(0.01)
    e.stop()
    assert e.wait(10000)
    qapp.processEvents()
    assert len(frames) >= 3 and (frames[0].width(), frames[0].height()) == (640, 360)
    assert {"fps", "vcam", "meeting", "gesture", "message", "scores", "invisible"} <= statuses[-1].keys()
    assert cap.released and e.cap is None


# --- recording a gesture -------------------------------------------------------------
def record(e, frame_size=(1280, 720)):
    """Run a recording to its end on simulated time; returns the status messages seen along the way."""
    e._do_record()
    t0, messages = time.perf_counter(), []
    for i in range(round((engine.RECORD_COUNTDOWN + engine.RECORD_SECONDS + 1) / DT)):
        e.step(BLACK(*frame_size), t0 + i * DT)
        messages.append(e._status(0.0)["message"])
    return messages


def test_recording_a_held_pose_counts_down_then_emits_a_template(tmp_path, qapp):
    e = make_engine(Store(tmp_path), tracker=FakeTracker([Hand(*hand3d(pick(2), False))]))
    got = []
    e.recorded.connect(got.append)
    messages = record(e)
    assert len(got) == 1 and got[0]["hands"] == 1 and got[0]["name"] == ""
    assert messages[0] == "record_countdown" and "recording" in messages and messages[-1] == ""
    assert e._status(0.0)["message_args"] == {}


def test_recording_with_no_hand_reports_no_hand(tmp_path, qapp):
    e = make_engine(Store(tmp_path))
    got = []
    e.recorded.connect(got.append)
    record(e)
    assert got == ["no_hand"]


def test_record_countdown_reports_the_seconds_left(tmp_path, qapp):
    e = make_engine(Store(tmp_path))
    e._do_record()
    t0 = time.perf_counter()
    e.step(BLACK(), t0)
    assert e._status(0.0)["message"] == "record_countdown" and e._status(0.0)["message_args"] == {"seconds": 3}
    e.step(BLACK(), t0 + 1.2)
    assert e._status(0.0)["message_args"] == {"seconds": 2}


# --- the room photo and the vanish -----------------------------------------------------
def test_a_fired_vanish_jurus_hides_you_once_the_room_is_photographed(tmp_path, qapp):
    store = Store(tmp_path)
    e = make_engine(store)
    e.cfg = {**e.cfg, "vanish_defaults": {**e.cfg["vanish_defaults"], "capture_delay": 0.1}}  # a quick photo
    t = time.perf_counter()
    e.step(np.full((720, 1280, 3), 90, np.uint8), t)
    e._do_capture_room()
    seen = []
    for i in range(1, 40):
        out, _ = e.step(np.full((720, 1280, 3), 90, np.uint8), t + i * DT)
        seen.append(e._status(0.0)["message"])
        if e.vanish.plate is not None:
            break
    assert "room_countdown" in seen and seen[-1] == "room_captured" and store.background.exists()

    e.tracker.hands = [VERTICAL, HORIZONTAL]  # the crossed-fingers seal
    t += 41 * DT
    for i in range(round(1.0 / DT)):          # "invisible" must come well within a second of the seal
        e.step(np.full((720, 1280, 3), 200, np.uint8), t + i * DT)
        if e._status(0.0)["invisible"]:
            break
    assert e._status(0.0)["invisible"]
    t += (i + 1) * DT
    while e.vanish.busy:
        t += DT
        e.step(np.full((720, 1280, 3), 200, np.uint8), t)
    out, _ = e.step(np.full((720, 1280, 3), 200, np.uint8), t + DT)
    assert (out == 90).all()                  # the meeting sees the empty room


def test_try_jurus_with_no_room_photo_explains(tmp_path, qapp):
    e = make_engine(Store(tmp_path))
    t = time.perf_counter()
    e.step(BLACK(), t)
    e._do_try_jurus("Menghilang")
    e.step(BLACK(), time.perf_counter())
    assert e._status(0.0)["message"] == "no_room_photo" and not e._status(0.0)["invisible"]
