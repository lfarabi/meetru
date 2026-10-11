import numpy as np
import pytest

from gestures import (FIST, FOUR, NO_HAND, ONE, OPEN_PALM, THREE, TWO, GestureClassifier, openness,
                      raised_fingers, thumb_out)
from tracking import Smoother, palm_center

CFG = {"fist_below": 0.55, "debounce_frames": 3, "lost_frames": 5,
       "finger_straight": 0.82, "thumb_out": 0.1}
MCP_X = [-0.3, -0.1, 0.1, 0.3]


def hand(tip_y: float, scale: float = 100.0) -> np.ndarray:
    """Synthetic upright hand: wrist at origin, MCPs at y=-1, fingertips at tip_y."""
    lm = np.zeros((21, 3), dtype=np.float32)
    for i, x in zip([5, 9, 13, 17], MCP_X):
        lm[i] = (x, -1, 0)
    for i, x in zip([8, 12, 16, 20], MCP_X):
        lm[i] = (x, tip_y, 0)
    return lm * scale


OPEN, CLOSED = hand(-2.2), hand(-0.6)


def test_palm_center_is_mean_of_wrist_and_mcps():
    assert np.allclose(palm_center(hand(-2)), [0, -80, 0], atol=1e-4)


def test_openness_is_scale_invariant():
    assert abs(openness(hand(-2.2, 50)) - openness(hand(-2.2, 400))) < 1e-5
    assert openness(OPEN) > 1.0 and openness(CLOSED) < CFG["fist_below"]


# 3D hand in meters, palm to the camera, fingers up (index on +x). Each finger raised or folded.
FINGER_MCP = {5: (0.03, -0.09), 9: (0.01, -0.095), 13: (-0.01, -0.09), 17: (-0.03, -0.08)}


def hand3d(raised=(True, True, True, True), thumb=True):
    """Returns (pixel landmarks, world landmarks); pixels are world × 1000 around (640, 500)."""
    w = np.zeros((21, 3), np.float32)
    for up, (mcp, (x, y)) in zip(raised, FINGER_MCP.items()):
        w[mcp] = (x, y, 0)
        if up:   # straight up
            w[mcp + 1:mcp + 4] = [(x, y - 0.04, 0), (x, y - 0.065, 0), (x, y - 0.085, 0)]
        else:    # curled toward the palm
            w[mcp + 1:mcp + 4] = [(x, y - 0.03, -0.02), (x, y - 0.01, -0.04), (x, y + 0.015, -0.03)]
    w[1], w[2] = (0.02, -0.02, 0), (0.04, -0.04, 0)
    w[3], w[4] = ((0.055, -0.06, 0), (0.08, -0.07, 0)) if thumb else ((0.02, -0.05, -0.02), (-0.01, -0.06, -0.02))
    return w * 1000 + np.array([640, 500, 0], np.float32), w


def pick(n):
    return tuple(i < n for i in range(4))


def test_raised_fingers_counts_each_finger():
    for n in range(5):
        assert raised_fingers(hand3d(pick(n))[1], CFG["finger_straight"]) == n


def test_thumb_out_separates_open_thumb_from_folded():
    assert thumb_out(hand3d(thumb=True)[1]) > 0.3
    assert thumb_out(hand3d(thumb=False)[1]) < -0.3


PALM = hand3d(thumb=True)
FIST_HAND = hand3d(pick(0), thumb=False)


def feed(c, h, n):
    for _ in range(n):
        g = c.update(*h) if h is not None else c.update(None, None)
    return g


def test_each_gesture_is_recognized():
    cases = {OPEN_PALM: PALM, FIST: FIST_HAND, ONE: hand3d(pick(1), False), TWO: hand3d(pick(2), False),
             THREE: hand3d(pick(3), False), FOUR: hand3d(pick(4), False)}
    for expected, h in cases.items():
        assert feed(GestureClassifier(CFG), h, 3) == expected, expected


def test_open_palm_tolerates_one_bent_finger_when_thumb_is_out():
    assert feed(GestureClassifier(CFG), hand3d(pick(3), thumb=True), 3) == OPEN_PALM


def test_one_raised_finger_is_not_a_fist_even_when_mean_openness_is_low():
    one_finger = (FIST_HAND[0], hand3d(pick(1), False)[1])  # fist-like openness, index straight in 3D
    assert feed(GestureClassifier(CFG), one_finger, 3) == ONE


def test_finger_count_ignores_an_out_thumb_below_three():
    assert feed(GestureClassifier(CFG), hand3d(pick(2), thumb=True), 3) == TWO


def test_gesture_changes_only_after_debounce():
    c = GestureClassifier(CFG)
    assert feed(c, PALM, 2) == NO_HAND
    assert feed(c, PALM, 1) == OPEN_PALM
    assert feed(c, FIST_HAND, 2) == OPEN_PALM
    assert feed(c, FIST_HAND, 1) == FIST


def test_single_frame_glitch_is_ignored():
    c = GestureClassifier(CFG)
    feed(c, PALM, 3)
    c.update(*FIST_HAND)
    assert feed(c, PALM, 3) == OPEN_PALM


def test_half_closed_hand_keeps_previous_gesture():
    c = GestureClassifier(CFG)
    feed(c, FIST_HAND, 3)
    half = (PALM[0], FIST_HAND[1])  # open enough not to be a fist, but no finger straight
    assert feed(c, half, 10) == FIST


def test_brief_hand_loss_does_not_report_no_hand():
    c = GestureClassifier(CFG)
    feed(c, PALM, 3)
    assert feed(c, None, 4) == OPEN_PALM
    assert feed(c, None, 1) == NO_HAND


def test_smoother_blends_and_resets_on_loss():
    s = Smoother(0.5)
    assert s.update(np.array([0.0])) == 0
    assert s.update(np.array([10.0])) == 5
    assert s.update(None) is None
    assert s.update(np.array([10.0])) == 10  # no glide from stale position


# --- orb state machine -------------------------------------------------------
from gestures import (ACTIVE, BLAST, BUILTIN_GESTURES, CHARGING, FADING, HIDDEN, RAISED_FIST, SEAL,  # noqa: E402
                      TWO_HAND_GESTURES, WOW, OrbMachine)

import yaml  # noqa: E402

M_CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
AIM, HOLD, MID = 0.92, -0.64, 0.6  # facing: palm to camera, palm up (holding), mid-turn (between thresholds)
DT = 0.04


class Driver:
    def __init__(self):
        self.m, self.t = OrbMachine(M_CFG), 0.0

    def run(self, gesture, facing, seconds):
        for _ in range(max(1, round(seconds / DT))):
            self.t += DT
            self.m.update(gesture, facing, self.t)
        return self.m.state


def held(seconds):
    """Orb summoned facing the camera, then held palm-up for `seconds`."""
    d = Driver()
    d.run(OPEN_PALM, AIM, 0.3)
    d.run(OPEN_PALM, HOLD, seconds)
    return d


def test_open_palm_spawns_and_fades_in():
    d = Driver()
    assert d.run(OPEN_PALM, AIM, 0.1) == ACTIVE
    assert 0 < d.m.presence < 1
    d.run(OPEN_PALM, AIM, 0.3)
    assert d.m.presence == 1


def test_facing_the_camera_does_not_charge():
    d = Driver()
    d.run(OPEN_PALM, AIM, 2.0)
    assert d.m.state == ACTIVE and d.m.charge == 0


def test_holding_palm_up_charges_and_saturates():
    d = held(1.5)
    assert d.m.state == CHARGING and 0.4 < d.m.charge < 0.6
    d.run(OPEN_PALM, HOLD, 3)
    assert d.m.charge == 1


def test_aiming_at_the_camera_after_charging_blasts():
    d = held(1.0)
    assert d.run(OPEN_PALM, AIM, DT) == BLAST
    assert d.m.blast_power > 0.3


def test_aiming_too_soon_just_keeps_the_orb():
    d = held(0.3)
    assert d.run(OPEN_PALM, AIM, 0.2) == ACTIVE and d.m.charge == 0


def test_mid_turn_keeps_charging_without_blasting():
    d = held(1.0)
    assert d.run(OPEN_PALM, MID, 0.4) == CHARGING


def test_thumb_misread_mid_turn_does_not_drop_the_charge():
    d = held(1.0)
    d.run(FOUR, MID, 0.2)
    assert d.m.state == CHARGING
    assert d.run(OPEN_PALM, AIM, DT) == BLAST


def test_fist_extinguishes_the_orb_audibly():
    d = held(1.0)
    assert d.run(FIST, None, DT) == FADING and not d.m.silent_fade


def test_blast_ends_hidden_with_cooldown_then_respawns():
    d = held(1.0)
    d.run(OPEN_PALM, AIM, DT)
    assert d.run(OPEN_PALM, AIM, M_CFG["blast"]["duration"]) == HIDDEN
    assert d.run(OPEN_PALM, AIM, 0.2) == HIDDEN  # still cooling down
    assert d.run(OPEN_PALM, AIM, 1.0) == ACTIVE


def test_hand_loss_fades_out_then_hides():
    d = Driver()
    d.run(OPEN_PALM, AIM, 0.5)
    assert d.run(NO_HAND, None, 0.1) == FADING
    assert d.run(NO_HAND, None, 0.6) == HIDDEN


def test_hand_returning_during_fade_revives_orb():
    d = Driver()
    d.run(OPEN_PALM, AIM, 0.5)
    d.run(NO_HAND, None, 0.1)
    assert d.run(OPEN_PALM, AIM, DT) == ACTIVE


def test_finger_gesture_fades_orb_silently():
    d = Driver()
    d.run(OPEN_PALM, AIM, 0.5)
    assert d.run(TWO, AIM, DT) == FADING and d.m.silent_fade
    d = Driver()
    d.run(OPEN_PALM, AIM, 0.5)
    d.run(NO_HAND, None, DT)
    assert not d.m.silent_fade  # losing the hand still powers down audibly


def test_finger_gesture_does_not_spawn_orb():
    assert Driver().run(THREE, AIM, 0.5) == HIDDEN


def test_orb_can_be_summoned_by_another_gesture():
    m = OrbMachine(M_CFG, trigger=TWO)
    m.update(TWO, AIM, 0.0)
    m.update(TWO, AIM, 0.1)
    assert m.state == ACTIVE
    m2 = OrbMachine(M_CFG, trigger=TWO)
    m2.update(OPEN_PALM, AIM, 0.0)
    assert m2.state == HIDDEN


def test_four_tolerance_only_for_open_palm_trigger():
    m = OrbMachine(M_CFG, trigger=TWO)
    for i in range(10):
        m.update(TWO, AIM, i * DT)
    for i in range(10, 20):
        m.update(TWO, HOLD, i * DT)
    assert m.state == CHARGING
    assert m.update(FOUR, MID, 20 * DT) == FADING     # the thumb tolerance belongs to the open palm only


def test_new_gesture_ids():
    assert (RAISED_FIST, SEAL, WOW) == ("RAISED_FIST", "SEAL", "WOW")
    assert BUILTIN_GESTURES == (ONE, TWO, THREE, FOUR, OPEN_PALM, FIST, RAISED_FIST, SEAL, WOW)
    assert TWO_HAND_GESTURES == {SEAL, WOW}


# --- audio mixer --------------------------------------------------------------
import audio  # noqa: E402
from audio import AudioEngine, Mixer, synthesize  # noqa: E402

SR = 48000
A_CFG = {**M_CFG["audio"], "sample_rate": SR}
SOUNDS_ = synthesize(SR)


def test_synthesized_sounds_are_clean():
    for name, x in SOUNDS_.items():
        assert x.dtype == np.float32 and np.isfinite(x).all() and 0.3 < np.abs(x).max() <= 0.91, name


def test_builtin_sound_names_match_the_synthesized_set():
    assert set(audio.BUILTIN_SOUNDS) == set(SOUNDS_)


def test_loops_are_seamless():
    hum = SOUNDS_["charge_loop"]
    assert abs(hum[-1] - hum[0]) < 0.05       # continuous hum wraps without a jump
    beat = SOUNDS_["finger_4"]
    assert np.abs(beat[-480:]).max() < 0.01   # beat has fully decayed before wrapping to the next hit


def test_mixer_is_silent_until_something_plays():
    mx = Mixer(SOUNDS_, A_CFG)
    assert not mx.render(256).any()
    mx.play("blast")
    assert np.abs(mx.render(256)).max() > 0


def test_one_shot_voice_is_released_when_finished():
    mx = Mixer(SOUNDS_, A_CFG)
    mx.play("activation")
    for _ in range(len(SOUNDS_["activation"]) // 256 + 2):
        mx.render(256)
    assert not mx._voices


def test_charge_loop_ramps_in_and_out_and_mute_silences():
    mx = Mixer(SOUNDS_, A_CFG)
    mx.set_loop("charge_loop", 1.0)
    first = np.abs(mx.render(256)).max()
    for _ in range(40):
        loud = np.abs(mx.render(256)).max()
    assert first < loud  # ramped in, no click
    mx.muted = True
    assert not mx.render(256).any()
    mx.muted = False
    mx.set_loop("charge_loop", 0.0)
    for _ in range(100):
        mx.render(256)
    assert np.abs(mx.render(256)).max() < 1e-3


def playing(mx):
    mx.render(256)
    return [v[0] for v in mx._voices if not v[4]]


def new_engine():
    return AudioEngine({**A_CFG, "enabled": False})


def test_sound_set_can_be_swapped_at_runtime():
    a = AudioEngine({**M_CFG["audio"], "enabled": False})
    a.set_sounds({"ding": np.ones(4800, np.float32)}, set())
    assert a.duration("ding") == pytest.approx(0.1) and "finger_1" in a.mixer.sounds
    a.play("ding"); assert np.abs(a.mixer.render(256)).max() > 0


def test_loop_sounds_follow_the_set():
    a = AudioEngine({**M_CFG["audio"], "enabled": False})
    a.set_sounds({"hum": np.ones(4800, np.float32)}, {"hum"})
    a.set_loop("hum", 1.0); assert np.abs(a.mixer.render(4096)).max() > 0


def test_swap_reaches_the_meeting_mixer_and_leaves_playing_voices_alone():
    e = new_engine()
    e.add_meeting_mixer(MEET_CFG)
    e.play("blast")
    e.mixer.render(256)
    old = e.mixer.sounds
    e.set_sounds({"blast": np.zeros(4800, np.float32), "ding": np.ones(4800, np.float32)}, set())
    assert e.mixer.sounds is not old and old["blast"] is e.builtin["blast"]   # a new dict; the old one wasn't edited
    assert e.meeting_mixer.sounds["ding"] is e.mixer.sounds["ding"]
    assert e.mixer._voices[0][1] is e.builtin["blast"]                         # the voice already playing keeps its buffer
    e.set_sounds({}, set())
    assert "ding" not in e.mixer.sounds and "ding" not in e.meeting_mixer.sounds
    assert e.duration("ding") == 0.0


def test_unknown_sound_names_are_ignored_not_fatal():
    mx = Mixer(SOUNDS_, A_CFG)
    mx.play("gone")
    mx.set_loop("gone", 1.0)
    assert not mx.render(256).any()


class SwappedAfterFirstLookup(dict):
    """A sound set that a set_sounds() swap replaces (here: empties) right after the first lookup."""

    def __init__(self, d):
        super().__init__(d)
        self.lookups = 0

    def _first(self):
        self.lookups += 1
        return self.lookups == 1

    def __contains__(self, k):
        return super().__contains__(k) and self._first()

    def __getitem__(self, k):
        if not self._first():
            raise KeyError(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        return super().get(k, default) if self._first() else default


def test_render_reads_the_sound_set_once_per_command():
    mx = Mixer(SOUNDS_, A_CFG)
    mx.sounds = SwappedAfterFirstLookup(SOUNDS_)  # a swap lands between a check and a use
    mx.play("blast")
    assert np.abs(mx.render(256)).max() > 0       # no KeyError on the audio thread, the sound plays


def test_charge_loop_survives_a_swap():
    e = new_engine()
    e.set_loop("charge_loop", 1.0)
    for _ in range(40):
        e.mixer.render(256)
    e.set_sounds({"ding": np.ones(480, np.float32)}, set())
    assert np.abs(e.mixer.render(256)).max() > 0.1  # still humming, no restart from silence


def test_stop_only_touches_the_named_sound():
    mx = Mixer(SOUNDS_, A_CFG)
    mx.play("blast")
    mx.stop("finger_3")
    mx.render(256)
    assert [v[0] for v in mx._voices if not v[4]] == ["blast"]


def test_orb_sound_cuts_a_finger_sound():
    e = new_engine()
    e.play("finger_3")
    assert playing(e.mixer) == ["finger_3"]
    e.update(HIDDEN, ACTIVE, 0.0, 0.0)  # open palm spawns the orb
    assert playing(e.mixer) == ["activation"]


def test_new_one_shot_fades_out_the_previous_one():
    mx = Mixer(SOUNDS_, A_CFG)
    mx.play("finger_3")
    mx.render(256)
    mx.play("finger_3")
    mx.render(256)  # old copy fades out during this block
    mx.render(256)
    assert [v[0] for v in mx._voices] == ["finger_3"] and mx._voices[0][2] == 512


# --- 3D palm normal, anchor and occlusion ------------------------------------
from orb import occlude_by_hand, orb_anchor  # noqa: E402
from tracking import palm_normal  # noqa: E402

# wrist/index/pinky plane facing the camera; which face is the palm depends on where the fingers curl
PLANE = np.zeros((21, 3), np.float32)
PLANE[5] = (0.03, -0.08, 0)    # index MCP
PLANE[17] = (-0.03, -0.07, 0)  # pinky MCP
CURL_IDX = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]


def curled(z: float) -> np.ndarray:
    w = PLANE.copy()
    w[CURL_IDX, 2] = z
    return w


def test_palm_faces_the_side_the_fingers_curl_to():
    n, side = palm_normal(curled(-0.01))  # fingers toward the camera: palm faces the camera
    assert n[2] < -0.99
    n, _ = palm_normal(curled(+0.01))     # fingers curl away: back of the hand faces the camera
    assert n[2] > 0.99


def test_flat_hand_keeps_previous_palm_side():
    _, side = palm_normal(curled(-0.01))
    n, _ = palm_normal(curled(0.0005), prev_side=side)
    assert n[2] < -0.99


ORB_CFG = {**M_CFG["orb"], "hover": 1.0, "palm_position": 0.5, "radius": 0.5, "center_lock": 0.75}


def test_orb_stays_on_palm_when_palm_faces_camera():
    (x, y), r = orb_anchor(OPEN, np.array([0, 0, -1.0]), ORB_CFG)
    assert abs(x) < 1e-3 and abs(y - (-50)) < 1e-3  # midway wrist (0) → knuckles (-100)


def test_small_tilt_toward_camera_keeps_orb_centered():
    tilted = np.array([0, 0.4, -0.92])
    (x, y), r = orb_anchor(OPEN, tilted / np.linalg.norm(tilted), ORB_CFG)
    assert abs(y - (-50)) < 0.1 * r


def test_orb_lifts_above_palm_facing_up():
    (x, y), r = orb_anchor(OPEN, np.array([0, -1.0, 0]), ORB_CFG)
    assert abs(r - 0.5 * 100.5) < 1 and abs(y - (-50 - r)) < 1e-3


def test_hand_covers_orb_only_where_the_hand_is():
    lm = hand(-2.2) + np.array([320, 400, 0], np.float32)  # wrist at (320, 400), fingers up
    original = np.zeros((720, 1280, 3), np.uint8)
    frame = np.full_like(original, 255)  # pretend the orb painted everything white
    occlude_by_hand(frame, original, lm, 1.0)
    assert frame[350, 320].max() == 0       # palm pixel restored from the original
    assert frame[100, 1000].min() == 255    # far away: orb untouched


# --- sound files --------------------------------------------------------------
import soundfile as sf  # noqa: E402


def test_mp3_is_loaded_mono_and_resampled(tmp_path):
    tone = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32) * 0.5
    sf.write(tmp_path / "beep.mp3", np.stack([tone, tone], axis=1), 44100)  # 1 s stereo at 44.1 kHz
    x = audio.load_sound(tmp_path / "beep.mp3", 48000)
    assert x.ndim == 1 and abs(len(x) - 48000) < 3000 and 0.3 < np.abs(x).max() < 0.7


# --- meeting audio (mic + SFX into VB-CABLE) -------------------------------------
from audio import Fifo  # noqa: E402

MEET_CFG = {"enabled": True, "mic_device": "x", "output_device": "y", "mic_volume": 1.0,
            "sfx_volume": 0.8, "max_latency_ms": 60}


def test_fifo_keeps_order_pads_underflow_and_caps_latency():
    f = Fifo(4)
    f.write(np.array([1, 2, 3], np.float32))
    assert list(f.read(2)) == [1, 2]
    assert list(f.read(3)) == [3, 0, 0]          # ran dry: silence, not a stall
    f.write(np.arange(10, dtype=np.float32))
    assert list(f.read(4)) == [6, 7, 8, 9]       # backed up: oldest dropped, latency stays bounded


def meeting_out(e, frames=256):
    out = np.zeros((frames, 2), np.float32)
    e._meeting_callback(out, frames, None, None)
    return out


def test_meeting_output_carries_mic_and_sfx():
    e = new_engine()
    e.add_meeting_mixer(MEET_CFG)
    mic = np.full(256, 0.1, np.float32)
    e._fifo.write(mic)
    assert np.allclose(meeting_out(e)[:, 0], 0.1)       # voice passes through untouched
    e.play("finger_1")                                   # SFX goes to both mixers
    assert e.meeting_mixer._queue.qsize() == 1
    loudest = 0.0
    for _ in range(20):  # ~0.1 s of the beep
        e._fifo.write(mic)
        out = meeting_out(e)
        assert np.allclose(out[:, 0], out[:, 1])
        loudest = max(loudest, float(np.abs(out[:, 0] - 0.1).max()))
    assert loudest > 0.05


def test_mute_silences_sfx_everywhere_but_not_the_voice():
    e = new_engine()
    e.add_meeting_mixer(MEET_CFG)
    e.muted = True
    e.play("finger_3")
    e._fifo.write(np.full(256, 0.1, np.float32))
    assert np.allclose(meeting_out(e)[:, 0], 0.1)
    assert not e.mixer.render(256).any()


def test_list_devices_filters_by_host_api_and_direction(monkeypatch):
    import sounddevice as sd
    monkeypatch.setattr(sd, "query_hostapis", lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}])
    monkeypatch.setattr(sd, "query_devices", lambda: [
        {"name": "Mic A", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0},
        {"name": "Speakers", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Mic MME", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0}])
    assert audio.list_devices("Windows WASAPI", "input") == ["Mic A"]
    assert audio.list_devices("Windows WASAPI", "output") == ["Speakers"]


# --- video orb ------------------------------------------------------------------
import cv2  # noqa: E402

from orb import EnergyOrb, load_video_sprite  # noqa: E402


def write_orb_clip(path, n=48, side=160, radius=40):
    """Black clip: 8 black frames, a disc growing for 8 frames, then a steady disc (like orb.mp4)."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (side, side))
    for i in range(n):
        f = np.zeros((side, side, 3), np.uint8)
        r = 0 if i < 8 else min(radius, int(radius * (i - 7) / 8))
        if r:
            cv2.circle(f, (side // 2, side // 2), r, (255, 230, 200), -1)
        out.write(f)
    out.release()


def test_video_sprite_finds_intro_loop_and_sphere_size(tmp_path):
    write_orb_clip(tmp_path / "orb.mp4")
    intro, loop, fps, sphere = load_video_sprite(tmp_path / "orb.mp4")
    assert 4 <= len(intro) <= 10 and len(loop) >= 20 and abs(fps - 24) < 1
    assert abs(sphere - 40 / 160) < 0.03
    assert loop[0].shape == (320, 320, 3) and loop[0][:4, :4].max() == 0  # vignette: corners stay black


def test_missing_orb_video_falls_back_to_procedural():
    cfg = {**M_CFG, "orb": {**M_CFG["orb"], "style": "video", "video": "assets/video/not-there.mp4"}}
    assert EnergyOrb(cfg).video is None


# --- vanishing jutsu --------------------------------------------------------------
import jutsu  # noqa: E402
from jutsu import Vanish, load_clip, seal_point, wow_pose  # noqa: E402
from tracking import Hand  # noqa: E402

TWO_UP_WORLD = hand3d(pick(2), False)[1]  # index + middle raised in 3D


def two_finger_hand(base_a, base_b, tip_a, tip_b):
    lm = np.zeros((21, 3), np.float32)
    lm[5, :2], lm[9, :2], lm[8, :2], lm[12, :2] = base_a, base_b, tip_a, tip_b
    return Hand(lm, TWO_UP_WORLD)


VERTICAL = two_finger_hand((600, 400), (620, 400), (600, 300), (620, 300))       # fingers pointing up
HORIZONTAL = two_finger_hand((700, 350), (700, 370), (560, 350), (560, 370))     # pointing sideways, across them


def test_crossed_two_finger_hands_make_the_seal():
    p = seal_point([VERTICAL, HORIZONTAL], 0.82)
    assert p is not None and 580 < p[0] < 660 and 330 < p[1] < 380


def test_parallel_or_apart_hands_are_not_the_seal():
    parallel = two_finger_hand((700, 400), (720, 400), (700, 300), (720, 300))
    far = two_finger_hand((1200, 650), (1200, 670), (1060, 650), (1060, 670))
    assert seal_point([VERTICAL, parallel], 0.82) is None
    assert seal_point([VERTICAL, far], 0.82) is None
    assert seal_point([VERTICAL], 0.82) is None


def test_greenscreen_clip_is_keyed_trimmed_and_cropped(tmp_path):
    path = tmp_path / "smoke.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    for i in range(24):
        f = np.zeros((180, 320, 3), np.uint8) if i == 23 else np.full((180, 320, 3), (35, 216, 1), np.uint8)
        if 6 <= i < 23:
            cv2.circle(f, (100, 90), 10 + 2 * (i - 6), (245, 245, 245), -1)
        out.write(f)
    out.release()
    clip = load_clip(path)
    frames, fps, peak, (ax, ay) = clip.frames, clip.fps, clip.peak, clip.anchor
    assert 15 <= len(frames) <= 17 and abs(fps - 30) < 1       # green lead-in and black tail dropped
    color, alpha = frames[peak]
    h, w = alpha.shape
    assert alpha[h // 2, w // 2] > 230 and alpha[0, 0] < 25     # smoke opaque, green gone
    assert color[h // 2, w // 2, 1] <= color[h // 2, w // 2, 2] + 5  # despilled: no green cast
    assert abs(ax - w / 2) < 3 and abs(ay - h / 2) < 3             # densest part = the disc's center


def vanish_cfg(tmp_path, smoke=None):
    return {"smoke": smoke, "background": str(tmp_path / "bg.png"), "capture_delay": 1, "smoke_height": 1.3}


SEAL_AT = np.array([640, 360], np.float32)


def run_vanish(v, t0, seconds):
    t = t0
    for _ in range(round(seconds / DT)):
        t += DT
        v.update(t)
    return t


def test_without_a_background_the_seal_explains_instead_of_vanishing(tmp_path):
    v = Vanish(vanish_cfg(tmp_path), (720, 1280, 3))
    v.fire(SEAL_AT, 0.0)
    t = run_vanish(v, 0.0, 1.0)
    assert v.mix == 0 and v.message(t) == "no_room_photo"


def test_an_unreadable_room_photo_counts_as_no_photo(tmp_path):
    (tmp_path / "bg.png").write_bytes(b"not a png")   # e.g. cut short by a crash mid-save
    v = Vanish(vanish_cfg(tmp_path), (720, 1280, 3))
    assert v.plate is None


def test_capture_then_seal_vanishes_and_seal_again_returns(tmp_path):
    v = Vanish(vanish_cfg(tmp_path), (720, 1280, 3))
    room = np.full((720, 1280, 3), 90, np.uint8)
    v.start_capture(0.0)
    v.capture_tick(room, 0.04)
    assert v.message(0.04) == "room_countdown" and v.countdown(0.04) == 1
    t = 0.0
    for _ in range(60):
        t += DT
        v.capture_tick(room, t)
    assert v.plate is not None and (tmp_path / "bg.png").exists()
    assert v.message(t) == "room_captured" and v.countdown(t) == 0
    v.fire(SEAL_AT, t)
    assert v.busy
    t = run_vanish(v, t, 1.0)
    assert v.mix == 1 and not v.busy         # gone
    v.fire(SEAL_AT, t)
    t = run_vanish(v, t, 1.0)
    assert v.mix == 0                        # back
    frame = np.zeros((720, 1280, 3), np.uint8)
    v.mix = 1
    v.apply_plate(frame)
    assert frame.mean() == 90                # the empty room replaces the camera image


def test_fire_while_the_vanish_is_running_is_ignored(tmp_path):
    v = Vanish(vanish_cfg(tmp_path), (720, 1280, 3))
    v.plate = np.full((720, 1280, 3), 90, np.uint8)
    v.fire(SEAL_AT, 0.0)
    t = run_vanish(v, 0.0, 0.35)
    assert v.mix > 0.5                       # past halfway: an unguarded fire would retarget to "back"
    v.fire(SEAL_AT, t)                       # a second seal mid-puff must not flip the target back
    t = run_vanish(v, t, 1.0)
    assert v.mix == 1


def test_smoke_is_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(jutsu, "_load", lambda *a: pytest.fail("no smoke configured, nothing to load"))
    v = Vanish(vanish_cfg(tmp_path, smoke=None), (720, 1280, 3))
    v.plate = np.full((720, 1280, 3), 90, np.uint8)
    v.fire(SEAL_AT, 0.0)
    assert 0 < run_vanish(v, 0.0, 0.25) and 0 < v.mix < 1   # plain crossfade, no clip
    run_vanish(v, 0.25, 0.5)
    assert v.mix == 1
    v.draw_smoke(np.zeros((720, 1280, 3), np.uint8), 0.1)   # nothing to draw, nothing breaks


def test_primary_hand_follows_the_same_hand_not_the_list_order():
    from tracking import pick_primary
    small, big = Hand(OPEN * 0.5 + 100, TWO_UP_WORLD), Hand(OPEN + 600, TWO_UP_WORLD)
    assert pick_primary([small, big], None) is big                          # nothing tracked yet: largest
    assert pick_primary([big, small], small.lm[0, :2] + 3) is small         # then whichever continues
    assert pick_primary([], None) is None


# --- jutsu sound ---------------------------------------------------------------------
def test_exclusive_sound_cannot_be_interrupted_and_silences_loops():
    mx = Mixer(SOUNDS_, A_CFG)
    mx.set_loop("charge_loop", 1.0)
    for _ in range(20):
        mx.render(256)                       # charge hum is running
    mx.play("finger_1", exclusive=True)      # the jutsu sound (a short beep stands in for it)
    mx.render(256)
    mx.play("blast")                         # orb / finger sounds while it plays...
    mx.play("finger_2")
    mx.render(256)
    assert [v[0] for v in mx._voices if not v[4]] == ["finger_1"]   # ...are dropped
    for _ in range(len(SOUNDS_["finger_1"]) // 256 + 2):
        out = mx.render(256)
    assert not mx._voices
    first_loop_block = np.abs(mx.render(256)).max()
    assert first_loop_block < 0.2                                    # hum returns ramping up, no click
    mx.play("blast")
    mx.render(256)
    assert [v[0] for v in mx._voices if not v[4]] == ["blast"]      # normal again afterwards


# --- wow pose ------------------------------------------------------------------------------
OPEN_WORLD4 = hand3d(pick(4), True)[1]


def open_hand_at(wrist, knuckle):
    """An open hand (4 fingers in 3D) whose wrist → middle knuckle runs from `wrist` to `knuckle`."""
    lm = np.zeros((21, 3), np.float32)
    lm[0, :2], lm[9, :2] = wrist, knuckle
    return Hand(lm, OPEN_WORLD4)


def test_open_hands_together_fingers_up_is_the_wow_pose():
    left, right = open_hand_at((600, 500), (560, 400)), open_hand_at((660, 500), (700, 400))
    assert wow_pose([left, right], 0.82)


def test_wow_pose_needs_hands_together_open_and_up():
    left = open_hand_at((600, 500), (560, 400))
    assert not wow_pose([left, open_hand_at((1000, 500), (1040, 400))], 0.82)   # apart
    assert not wow_pose([left, open_hand_at((660, 500), (760, 500))], 0.82)    # pointing sideways
    closed = Hand(left.lm, hand3d(pick(0), False)[1])
    assert not wow_pose([left, closed], 0.82)                                   # one fist
    assert not wow_pose([left], 0.82)


def write_dull_green_clip(path, n=30):
    """Like wow.mp4: a duller green (40,152,26) with something skin-coloured rising from below."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    for i in range(n):
        f = np.full((180, 320, 3), (40, 152, 26), np.uint8)
        if 5 <= i < 25:
            cv2.rectangle(f, (200, 180 - 6 * (i - 4)), (300, 180), (140, 170, 220), -1)
        out.write(f)
    out.release()


def test_dull_green_screen_keys_cleanly_and_keeps_its_framing(tmp_path):
    write_dull_green_clip(tmp_path / "wow.mp4")
    clip = load_clip(tmp_path / "wow.mp4")
    _, alpha = clip.frames[0]                                 # subject just rising from the bottom
    assert alpha[0, 0] < 25 and alpha[-1, alpha.shape[1] // 2] > 230   # green keyed out, subject kept
    assert clip.source_width == 160 and clip.origin[0] >= 95  # crop remembers it sat on the right


# --- raised fist → Figure ---------------------------------------------------------------------
from jutsu import raised_fist  # noqa: E402


def test_raised_fist_needs_the_fist_up_high():
    lm = np.zeros((21, 3), np.float32)
    lm[0, 1] = 300                                         # wrist 300 px down a 720 px frame
    assert raised_fist(FIST, lm, 720, 0.6)
    assert not raised_fist(OPEN_PALM, lm, 720, 0.6)
    lm[0, 1] = 600                                         # fist down by the desk
    assert not raised_fist(FIST, lm, 720, 0.6)
    assert not raised_fist(FIST, None, 720, 0.6)


def test_hold_trigger_rides_out_short_tracker_dropouts():
    from jutsu import HoldTrigger

    def run(pattern, gap):
        trig, t, fired = HoldTrigger(0.3, gap), 0.0, 0
        for active, seconds in pattern:
            for _ in range(round(seconds / DT)):
                t += DT
                fired += trig.update(active, t)
        return fired
    flicker = [(True, 0.2), (False, 0.12), (True, 0.2)]       # a hand hidden for 3 frames mid-seal
    assert run(flicker, gap=0.3) == 1                         # still one continuous hold
    assert run(flicker, gap=0.0) == 0                         # without the gap it never completes
    assert run([(True, 0.4), (False, 0.5), (True, 0.4)], gap=0.3) == 2   # a real let-go re-arms


def write_clip(path, background, n=24, subject=(245, 245, 245)):
    """A disc growing over `background` (a BGR colour, or a function of the frame index giving a frame)."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
    for i in range(n):
        f = background(i) if callable(background) else np.full((180, 320, 3), background, np.uint8)
        if 4 <= i < n - 2:
            cv2.circle(f, (160, 90), 10 + 2 * (i - 4), subject, -1)
        out.write(f)
    out.release()


def test_blue_screen_is_keyed_like_green(tmp_path):
    write_clip(tmp_path / "blue.mp4", (230, 60, 20))                       # BGR: a blue screen
    clip = load_clip(tmp_path / "blue.mp4")
    color, alpha = clip.frames[0]
    h, w = alpha.shape
    assert alpha[h // 2, w // 2] > 230 and alpha[0, 0] < 25                # subject kept, blue gone
    assert color[h // 2, w // 2, 0] <= color[h // 2, w // 2, 2] + 5        # no blue cast left on it


def test_black_background_is_keyed_by_distance(tmp_path):
    write_clip(tmp_path / "black.mp4", (4, 4, 4), subject=(60, 170, 255))  # orange glow on black
    clip = load_clip(tmp_path / "black.mp4")
    color, alpha = clip.frames[clip.peak]
    h, w = alpha.shape
    assert alpha[h // 2, w // 2] > 230 and alpha[0, 0] < 25


def test_clip_without_a_one_colour_background_is_shown_unedited(tmp_path):
    rng = np.random.default_rng(0)
    room = np.clip(np.linspace(40, 220, 320)[None, :, None] + rng.normal(0, 25, (180, 320, 3)), 0, 255).astype(np.uint8)
    write_clip(tmp_path / "plain.mp4", lambda i: room.copy(), n=12)       # an ordinary video: no flat backdrop
    clip = load_clip(tmp_path / "plain.mp4")
    color, alpha = clip.frames[0]
    assert len(clip.frames) == 12 and alpha.min() == 255                   # every frame, fully opaque
    assert alpha.shape == (90, 160) and clip.origin == (0, 0)              # whole frame, not cropped or keyed
