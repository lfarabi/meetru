import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from test_logic import HORIZONTAL, M_CFG, VERTICAL, hand3d, pick

from gestures import FIST, FOUR, ONE, OPEN_PALM, RAISED_FIST, SEAL, THREE, TWO, WOW
from poses import make_template
from rules import Rules, active_ids, build, hands_center
from store import Store
from tracking import Hand, palm_center


class FakeAudio:
    def __init__(self):
        self.calls = []

    def play(self, name, gain=1.0, exclusive=False):
        self.calls.append(("play", name, gain, exclusive))

    def stop(self, name):
        self.calls.append(("stop", name))

    def set_loop(self, name, gain, rate=1.0):
        self.calls.append(("loop", name, gain))

    def duration(self, name):
        return 1.0


def snd(n=1, mode="once", volume=1.0, file=None):
    return {"file": file or f"builtin:finger_{n}", "mode": mode, "volume": volume}


def J(name, gesture, **kw) -> dict:
    """A jurus dict with a finger_1 sound and hold 0.3; `kw` overrides, and None drops a key."""
    cfg = {"name": name, "enabled": True, "gesture": gesture, "hold": 0.3, "exclusive": False,
           "sound": snd(), **kw}
    return {k: v for k, v in cfg.items() if v is not None}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


@pytest.fixture
def rig(store):
    def make(*cfgs):
        jurus, _, _ = build(list(cfgs), store, [], 48000)
        audio = FakeAudio()
        return audio, Rules(jurus, audio)
    return make


def run(r, active, times, two_hands=False, blocked=False):
    """Every jurus that fired over the frames at `times`."""
    return [j for t in times for j in r.update(active, t, two_hands, blocked)]


def test_fires_once_per_hold_and_rearms_after_release(rig):
    a, r = rig(J("one", ONE))
    fired = [r.update({ONE}, t, False, False) for t in np.arange(0, 1.0, 0.04)]
    assert sum(map(bool, fired)) == 1
    r.update(set(), 1.5, False, False)
    assert [x for t in np.arange(1.6, 2.2, 0.04) for x in r.update({ONE}, t, False, False)]


def test_release_stops_a_non_exclusive_sound_and_new_jurus_cuts_old(rig):
    a, r = rig(J("one", ONE, sound=snd(1)), J("two", TWO, sound=snd(2)))
    run(r, {ONE}, np.arange(0, 0.5, 0.04))
    run(r, {TWO}, np.arange(0.5, 1.0, 0.04))        # straight from one gesture to the next
    assert a.calls == [("play", "finger_1", 1.0, False), ("stop", "finger_1"), ("play", "finger_2", 1.0, False)]
    r.update(set(), 1.0, False, False)
    assert a.calls[-1] == ("stop", "finger_2")


def test_exclusive_blocks_others_until_its_sound_ends(rig):
    a, r = rig(J("wow", WOW, exclusive=True, sound=snd(2)), J("one", ONE, sound=snd(1)))
    assert run(r, {WOW}, [0.0, 0.1, 0.2, 0.3])[0].cfg["name"] == "wow"          # fires at 0.3, sound lasts to 1.3
    assert run(r, {ONE}, [0.4, 0.5, 0.7, 1.0, 1.2]) == []                       # ONE has been held long enough
    assert [j.cfg["name"] for j in run(r, {ONE}, [1.4])] == ["one"]
    assert a.calls == [("play", "finger_2", 1.0, True), ("play", "finger_1", 1.0, False)]   # never stopped


def test_two_hand_grace_blocks_one_hand_jurus_but_not_two_hand(rig):
    a, r = rig(J("one", ONE), J("wow", WOW, sound=snd(2)))
    both = {ONE, WOW}
    assert [j.cfg["name"] for j in run(r, both, [0.0, 0.2, 0.4], two_hands=True)] == ["wow"]
    assert [j.cfg["name"] for j in run(r, both, [0.5])] == ["one"]               # fires once the grace is over


def test_blocked_stops_an_already_firing_sound(rig):
    a, r = rig(J("one", ONE, sound=snd(1)), J("four", FOUR, sound=snd(4, "loop")))
    both = {ONE, FOUR}
    run(r, both, [0.0, 0.2, 0.4])
    assert a.calls == [("play", "finger_1", 1.0, False), ("loop", "finger_4", 1.0)]
    assert run(r, both, [0.5, 0.6], blocked=True) == []                          # e.g. the orb is charging
    assert a.calls[2:] == [("stop", "finger_1"), ("loop", "finger_4", 0.0)]
    assert run(r, both, [0.8, 1.0]) == []                                        # not re-armed until released
    assert len(a.calls) == 4


def test_loop_mode_plays_while_held(rig):
    a, r = rig(J("four", FOUR, sound=snd(4, "loop", 0.5)))
    run(r, {FOUR}, np.arange(0, 1.0, 0.04))
    assert a.calls == [("loop", "finger_4", 0.5)]
    r.update(set(), 1.0, False, False)
    assert a.calls == [("loop", "finger_4", 0.5), ("loop", "finger_4", 0.0)]


def test_vanish_jurus_is_returned_when_fired(rig):
    a, r = rig(J("vanish", SEAL, exclusive=True, visual={"type": "vanish", "file": "media/smoke.mp4"}))
    (j,) = run(r, {SEAL}, [0.0, 0.2, 0.4])
    assert j.kind == "vanish" and j.overlay is None and j.problem is None and j.hands == 2
    assert a.calls == [("play", "finger_1", 1.0, True)]


def test_problems_are_flagged_and_never_fire(store, rig):
    sf.write(str(store.media / "empty.wav"), np.zeros(0, np.float32), 48000)
    (store.media / "bad.mp3").write_bytes(b"junk")
    (store.media / "bad.png").write_bytes(b"junk")
    cfgs = [J("g", "rec:gone"), J("s", ONE, sound=snd(file="media/bad.mp3")),
            J("v", TWO, visual={"type": "image", "file": "media/bad.png", "duration": "held"}),
            J("missing", FIST, sound=snd(file="media/nope.mp3")), J("empty", FOUR, sound=snd(file="media/empty.wav")),
            J("builtin", RAISED_FIST, sound=snd(file="builtin:nope")),
            J("both", OPEN_PALM, sound=snd(file="media/bad.mp3"), visual={"type": "image"}),
            J("typo", "FIVE"), J("off", SEAL, enabled=False), J("healthy", THREE, sound=snd(3))]
    jurus, _, _ = build(cfgs, store, [], 48000)
    assert [j.problem for j in jurus] == ["gesture", "sound", "visual", "sound", "sound", "sound",
                                          "sound", "gesture", None, None]
    a, r = rig(*cfgs)
    fired = run(r, {c["gesture"] for c in cfgs}, [0.0, 0.2, 0.4, 0.6])
    assert [j.cfg["name"] for j in fired] == ["healthy"]
    assert a.calls == [("play", "finger_3", 1.0, False)]


def test_recorded_match_replaces_the_finger_count():
    lm, w = hand3d(pick(2), False)
    h = Hand(lm, w)
    peace = make_template("peace", [[h]] * 10)
    assert active_ids(TWO, [h], h, h.lm, 720, M_CFG, [peace]) == {"rec:peace"}
    assert active_ids(TWO, [h], h, h.lm, 720, M_CFG, []) == {TWO}
    lm4, w4 = hand3d(pick(4), True)
    other = Hand(lm4, w4)
    assert active_ids(OPEN_PALM, [other], other, other.lm, 720, M_CFG, [peace]) == {OPEN_PALM}


def test_raised_fist_replaces_fist():
    h = Hand(*hand3d(pick(0), False))        # wrist at y = 500
    assert active_ids(FIST, [h], h, h.lm, 1000, M_CFG, []) == {RAISED_FIST}
    assert active_ids(FIST, [h], h, h.lm, 720, M_CFG, []) == {FIST}


def test_seal_wow_and_two_hand_recordings_are_added_to_the_gesture():
    assert active_ids(TWO, [VERTICAL, HORIZONTAL], VERTICAL, VERTICAL.lm, 720, M_CFG, []) == {TWO, SEAL}
    a = Hand(*hand3d(pick(4), True))          # the same open hand twice: wrists together, fingers up
    assert active_ids(OPEN_PALM, [a, a], a, a.lm, 720, M_CFG, []) == {OPEN_PALM, WOW}
    one, pair = make_template("open", [[a]] * 10), make_template("pair", [[a, a]] * 10)
    # a one-hand and a two-hand recording both match: they must not compete for one slot
    assert active_ids(OPEN_PALM, [a, a], a, a.lm, 720, M_CFG, [one, pair]) == {"rec:open", "rec:pair", WOW}


def test_hands_center_is_the_mean_of_the_palm_centres():
    a, b = Hand(*hand3d()), Hand(hand3d()[0] + np.array([200, 100, 0], np.float32), hand3d()[1])
    assert np.allclose(hands_center([a, b]), (palm_center(a.lm) + palm_center(b.lm)) / 2)
    assert hands_center([]) is None


def test_build_returns_sound_keys_loops_and_orb_sounds(store, rig):
    sf.write(str(store.media / "boom.wav"), np.zeros(4800, np.float32) + 0.1, 24000)
    cfgs = [J("orb", OPEN_PALM, sound=None, visual={"type": "orb"}, orb_sounds={"blast": "media/boom.wav"}),
            J("loop", FOUR, sound=snd(file="media/boom.wav", mode="loop")), J("one", ONE)]
    jurus, sounds, loops = build(cfgs, store, [], 48000)
    assert set(sounds) == {"blast", "media/boom.wav"} and len(sounds["blast"]) == 9600      # resampled to 48 kHz
    assert loops == {"media/boom.wav"}
    assert [j.sound for j in jurus] == [None, "media/boom.wav", "finger_1"]
    assert jurus[0].kind == "orb" and jurus[0].overlay is None

    a, r = rig(*cfgs)
    assert r.orb_jurus.cfg["name"] == "orb"
    assert run(r, {OPEN_PALM}, [0.0, 0.2, 0.4]) == [] and a.calls == []         # the engine drives the orb
    assert rig(J("orb", OPEN_PALM, sound=None, enabled=False, visual={"type": "orb"}))[1].orb_jurus is None
    assert rig(J("one", ONE))[1].orb_jurus is None


def test_try_fire_plays_once_and_blocks_like_a_fire(rig):
    a, r = rig(J("four", FOUR, sound=snd(4, "loop", 0.5)), J("x", WOW, exclusive=True, sound=snd(2)),
               J("one", ONE), J("bad", TWO, sound=snd(file="builtin:nope")))
    assert r.try_fire("nobody", 0.0) is None and r.try_fire("bad", 0.0) is None
    assert r.try_fire("four", 0.0).cfg["name"] == "four"
    r.update(set(), 0.1, False, False)                                           # no gesture, yet nothing is cut
    assert a.calls == [("play", "finger_4", 0.5, False)]                         # played, not looped
    assert r.try_fire("x", 1.0).cfg["name"] == "x"
    assert r.update(set(), 1.1, False, False) == [] and ("stop", "finger_2") not in a.calls
    assert run(r, {ONE}, [1.2, 1.5, 1.9]) == []                                  # blocked until x's sound ends at 2.0
    assert [j.cfg["name"] for j in run(r, {ONE}, [2.1])] == ["one"]


IMAGE = {"type": "image", "file": "media/dot.png", "position": "center", "size": 0.5}


@pytest.fixture
def dot(store):
    Image.new("RGBA", (20, 20), (255, 0, 0, 255)).save(store.media / "dot.png")


def test_visual_is_drawn_while_held_and_fades_after_release(dot, rig):
    a, r = rig(J("pic", ONE, sound=None, visual={**IMAGE, "duration": "held"}))
    frame = np.zeros((200, 200, 3), np.uint8)
    r.draw(frame, 0.0, None, True)
    assert not frame.any()
    run(r, {ONE}, [0.0, 0.2, 0.4])
    r.draw(frame, 0.4, None, True)
    assert frame.any()
    r.update(set(), 0.5, False, False)
    assert r.jurus[0].overlay.busy
    r.update(set(), 0.8, False, False)
    assert not r.jurus[0].overlay.busy
    frame[:] = 0
    r.draw(frame, 0.8, None, True)
    assert not frame.any()
    # the editor's Try button: a held visual shows for 2 s without needing a gesture
    r.try_fire("pic", 5.0)
    assert r.jurus[0].overlay.busy
    r.update(set(), 6.9, False, False)
    assert r.jurus[0].overlay.busy
    r.update(set(), 7.1, False, False)
    assert not r.jurus[0].overlay.busy


def test_exclusive_waits_for_its_visual_as_well(dot, rig):
    a, r = rig(J("big", TWO, exclusive=True, sound=None, visual={**IMAGE, "duration": 2.0}), J("one", ONE))
    assert [j.cfg["name"] for j in run(r, {TWO}, [0.0, 0.3])] == ["big"]
    assert run(r, {ONE}, [0.5, 1.0, 2.0, 2.2]) == []                             # the picture is up until 2.3
    assert [j.cfg["name"] for j in run(r, {ONE}, [2.4])] == ["one"]
