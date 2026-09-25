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

