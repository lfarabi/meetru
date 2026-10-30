# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 lfarabi
import cv2
import numpy as np
import pytest
from test_logic import hand3d, pick

from poses import RADIUS_FLOOR, best_match, distance, hand_descriptor, make_template, score
from tracking import Hand


def rotate(world, rvec):
    """Rotate the hand about its wrist, like turning it in front of the camera."""
    r = cv2.Rodrigues(np.array(rvec, np.float64))[0]
    return (world - world[0]) @ r.T + world[0]


def frames_of(world, n=30, noise=0.0015):
    """n single-hand frames with Gaussian jitter (metres) on the world landmarks."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        w = (world + rng.normal(0, noise, world.shape)).astype(np.float32)
        out.append([Hand(w * 1000 + np.array([640, 500, 0], np.float32), w)])
    return out


def test_descriptor_ignores_rotation_scale_and_mirroring():
    _, w = hand3d(pick(2), False)
    base = hand_descriptor(w)
    for v in (rotate(w, (0.4, -0.7, 1.1)), w * 1.3, w * np.array([-1, 1, 1], np.float32)):
        assert np.allclose(hand_descriptor(v), base, atol=1e-4)


def test_recorded_pose_matches_itself_rotated_and_rejects_another_pose():
    t = make_template("peace", frames_of(hand3d(pick(2), False)[1]))
    lm, w = hand3d(pick(2), False)
    rot = Hand(lm, rotate(w, (0.3, 0.5, -0.4)))
    assert best_match([t], [rot], rot) == "peace" and score(t, [rot], rot) > 0
    lm4, w4 = hand3d(pick(4), True)
    other = Hand(lm4, w4)
    assert best_match([t], [other], other) is None


def test_still_recording_still_matches_thanks_to_radius_floor():
    t = make_template("x", frames_of(hand3d(pick(1), False)[1], noise=0.0))
    assert t["radius"] == RADIUS_FLOOR


def test_two_hand_template_matches_in_either_order_and_needs_two_hands():
    a = Hand(*hand3d(pick(2), False))
    lm4, w4 = hand3d(pick(4), True)
    b = Hand(lm4 + np.array([150, 0, 0], np.float32), w4)
    t = make_template("duo", [[a, b]] * 30)
    assert best_match([t], [b, a], b) == "duo" and distance(t, [a], a) is None


def test_two_hand_template_cares_how_far_apart_the_hands_are():
    a = Hand(*hand3d(pick(2), False))
    lm4, w4 = hand3d(pick(4), True)

    def apart(px):
        return [a, Hand(lm4 + np.array([px, 0, 0], np.float32), w4)]

    t = make_template("duo", [apart(150)] * 30)
    near = apart(150)
    assert best_match([t], near, near[0]) == "duo"
    for px in (250, 400):  # 250 is only one palm further apart: unweighted, that error stays inside the radius
        far = apart(px)
        assert best_match([t], far, far[0]) is None and score(t, far, far[0]) <= 0, px


def test_jittery_recording_gets_a_wider_radius_than_the_floor():
    t = make_template("x", frames_of(hand3d(pick(2), False)[1], noise=0.01))
    assert t["radius"] > RADIUS_FLOOR


def test_recording_without_a_steady_hand_is_refused():
    with pytest.raises(ValueError, match="no_hand"):
        make_template("x", [[]] * 30)
    h = frames_of(hand3d()[1], n=1)[0][0]
    with pytest.raises(ValueError, match="unsteady"):
        make_template("x", [[h]] * 10 + [[h, h]] * 10 + [[]] * 10)
