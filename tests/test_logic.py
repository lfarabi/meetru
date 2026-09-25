import numpy as np

from gestures import FIST, NO_HAND, OPEN_PALM, GestureClassifier, openness
from tracking import Smoother, palm_center

CFG = {"fist_below": 0.9, "open_above": 1.2, "debounce_frames": 3, "lost_frames": 5}
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


def test_smoother_blends_and_resets_on_loss():
    s = Smoother(0.5)
    assert s.update(np.array([0.0])) == 0
    assert s.update(np.array([10.0])) == 5
    assert s.update(None) is None
    assert s.update(np.array([10.0])) == 10  # no glide from stale position


def feed(c, lm, n):
    for _ in range(n):
        g = c.update(lm)
    return g


def test_gesture_changes_only_after_debounce():
    c = GestureClassifier(CFG)
    assert feed(c, OPEN, 2) == NO_HAND
    assert feed(c, OPEN, 1) == OPEN_PALM
    assert feed(c, CLOSED, 2) == OPEN_PALM
    assert feed(c, CLOSED, 1) == FIST


def test_single_frame_glitch_is_ignored():
    c = GestureClassifier(CFG)
    feed(c, OPEN, 3)
    c.update(CLOSED)
    assert feed(c, OPEN, 3) == OPEN_PALM


def test_dead_zone_keeps_previous_gesture():
    c = GestureClassifier(CFG)
    feed(c, CLOSED, 3)
    middle = hand(-1.85)
    assert CFG["fist_below"] < openness(middle) < CFG["open_above"]
    assert feed(c, middle, 10) == FIST


def test_brief_hand_loss_does_not_report_no_hand():
    c = GestureClassifier(CFG)
    feed(c, OPEN, 3)
    assert feed(c, None, 4) == OPEN_PALM
    assert feed(c, None, 1) == NO_HAND



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


# --- audio mixer --------------------------------------------------------------
from audio import Mixer, synthesize  # noqa: E402

SR = 48000
A_CFG = {**M_CFG["audio"], "sample_rate": SR}
SOUNDS_ = synthesize(SR)


def test_synthesized_sounds_are_clean():
    for name, x in SOUNDS_.items():
        assert x.dtype == np.float32 and np.isfinite(x).all() and 0.3 < np.abs(x).max() <= 0.91, name


def test_charge_loop_is_seamless():
    loop = SOUNDS_["charge_loop"]
    assert abs(loop[-1] - loop[0]) < 0.05


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
    mx.loop_gain_target = 1.0
    first = np.abs(mx.render(256)).max()
    for _ in range(40):
        loud = np.abs(mx.render(256)).max()
    assert first < loud  # ramped in, no click
    mx.muted = True
    assert not mx.render(256).any()
    mx.muted, mx.loop_gain_target = False, 0.0
    for _ in range(60):
        mx.render(256)
    assert np.abs(mx.render(256)).max() < 1e-3
