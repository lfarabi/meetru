import numpy as np

from gestures import (FIST, FOUR, NO_HAND, ONE, OPEN_PALM, THREE, TWO, GestureClassifier, openness,
                      raised_fingers, thumb_out)
from tracking import Smoother, palm_center

CFG = {"fist_below": 0.55, "open_above": 0.8, "debounce_frames": 3, "lost_frames": 5,
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
    assert openness(OPEN) > CFG["open_above"] and openness(CLOSED) < CFG["fist_below"]


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
from gestures import ACTIVE, BLAST, CHARGING, FADING, HIDDEN, OrbMachine  # noqa: E402

import yaml  # noqa: E402

M_CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
FIST_O, MID_O, OPEN_O = 0.3, 0.7, 1.1  # openness values: fist, dead zone, open
DT = 0.04


class Driver:
    def __init__(self):
        self.m, self.t = OrbMachine(M_CFG), 0.0

    def run(self, gesture, openness, seconds):
        for _ in range(max(1, round(seconds / DT))):
            self.t += DT
            self.m.update(gesture, openness, self.t)
        return self.m.state


def charged(seconds):
    d = Driver()
    d.run(OPEN_PALM, OPEN_O, 0.3)
    d.run(FIST, FIST_O, seconds)
    return d


def test_open_palm_spawns_and_fades_in():
    d = Driver()
    assert d.run(OPEN_PALM, OPEN_O, 0.1) == ACTIVE
    assert 0 < d.m.presence < 1
    d.run(OPEN_PALM, OPEN_O, 0.3)
    assert d.m.presence == 1


def test_fist_charges_and_saturates():
    d = charged(1.5)
    assert d.m.state == CHARGING and 0.4 < d.m.charge < 0.6
    d.run(FIST, FIST_O, 3)
    assert d.m.charge == 1


def test_short_charge_release_returns_to_active():
    d = charged(0.3)
    assert d.run(OPEN_PALM, OPEN_O, 0.1) == ACTIVE


def test_quick_release_after_charge_blasts_immediately():
    d = charged(1.0)
    # raw openness jumps to open before the debounced gesture has caught up
    assert d.run(FIST, OPEN_O, 2 * DT) == BLAST
    assert d.m.blast_power > 0.3


def test_slow_release_does_not_blast():
    d = charged(1.0)
    d.run(FIST, MID_O, 0.4)  # lingering half-open longer than the release window
    assert d.run(OPEN_PALM, OPEN_O, 0.1) == ACTIVE


def test_single_open_glitch_does_not_blast():
    d = charged(1.0)
    d.run(FIST, OPEN_O, DT)
    assert d.run(FIST, FIST_O, 0.2) == CHARGING


def test_blast_ends_hidden_with_cooldown_then_respawns():
    d = charged(1.0)
    d.run(FIST, OPEN_O, 2 * DT)
    assert d.run(OPEN_PALM, OPEN_O, M_CFG["blast"]["duration"]) == HIDDEN
    assert d.run(OPEN_PALM, OPEN_O, 0.2) == HIDDEN  # still cooling down
    assert d.run(OPEN_PALM, OPEN_O, 1.0) == ACTIVE


def test_hand_loss_fades_out_then_hides():
    d = Driver()
    d.run(OPEN_PALM, OPEN_O, 0.5)
    assert d.run(NO_HAND, None, 0.1) == FADING
    assert d.run(NO_HAND, None, 0.6) == HIDDEN


def test_hand_returning_during_fade_revives_orb():
    d = Driver()
    d.run(OPEN_PALM, OPEN_O, 0.5)
    d.run(NO_HAND, None, 0.1)
    assert d.run(OPEN_PALM, OPEN_O, DT) == ACTIVE


def test_finger_gesture_fades_orb_silently():
    d = Driver()
    d.run(OPEN_PALM, OPEN_O, 0.5)
    assert d.run(TWO, 0.6, DT) == FADING and d.m.silent_fade
    d = Driver()
    d.run(OPEN_PALM, OPEN_O, 0.5)
    d.run(NO_HAND, None, DT)
    assert not d.m.silent_fade  # losing the hand still powers down audibly


def test_finger_gesture_does_not_spawn_orb():
    assert Driver().run(THREE, 0.9, 0.5) == HIDDEN


# --- audio mixer --------------------------------------------------------------
from audio import AudioEngine, Mixer, synthesize  # noqa: E402

SR = 48000
A_CFG = {**M_CFG["audio"], "sample_rate": SR}
SOUNDS_ = synthesize(SR)


def test_synthesized_sounds_are_clean():
    for name, x in SOUNDS_.items():
        assert x.dtype == np.float32 and np.isfinite(x).all() and 0.3 < np.abs(x).max() <= 0.91, name


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


FINGER_CFG = M_CFG["finger_sfx"]


def engine():
    return AudioEngine({**A_CFG, "enabled": False}, FINGER_CFG)  # no device: drive the mixer directly


def drain(e, blocks=1):
    return max(np.abs(e.mixer.render(256)).max() for _ in range(blocks))


def test_finger_once_sound_plays_on_entering_the_gesture_only():
    e = engine()
    e.update_fingers(NO_HAND, TWO, 1.0)
    assert drain(e) > 0
    for _ in range(200):
        e.mixer.render(256)  # let it finish
    e.update_fingers(TWO, TWO, 1.1)  # holding the gesture does not replay
    assert drain(e) == 0


def test_finger_once_sound_is_rate_limited():
    e = engine()
    e.update_fingers(NO_HAND, ONE, 1.0)
    e.update_fingers(ONE, NO_HAND, 1.05)
    e.update_fingers(NO_HAND, ONE, 1.1)  # flicker within the cooldown
    assert e.mixer._queue.qsize() == 1


def test_finger_loop_plays_while_held_and_stops_after():
    e = engine()
    e.update_fingers(NO_HAND, FOUR, 1.0)
    assert drain(e, 40) > 0.1
    e.update_fingers(FOUR, NO_HAND, 2.0)
    drain(e, 80)
    assert drain(e) < 1e-3


def test_missing_custom_sound_is_dropped_not_fatal():
    e = AudioEngine({**A_CFG, "enabled": False}, {**FINGER_CFG, "ONE": {"sound": "nope", "mode": "once", "volume": 1}})
    assert "ONE" not in e.fingers and "TWO" in e.fingers


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
