"""The jurus runtime: turns the user's jurus list into objects (sounds loaded, visuals built) and
decides every frame which of them fire.

Behaviour per jurus: it fires once per hold of its gesture and re-arms after release; a one-hand jurus
waits while two hands are about to pose; nothing fires while the orb or a vanish is busy or while
another *exclusive* jurus is still playing its sound or showing its picture. Letting go stops a
non-exclusive sound (an exclusive one plays to the end) and fades the visual; a tracker dropout shorter
than the trigger's gap doesn't count as letting go. A jurus whose gesture,
sound or visual can't be used is kept, flagged in `problem`, and never fires."""
import logging
import math
from dataclasses import dataclass

import numpy as np

import poses
from audio import BUILTIN_SOUNDS, load_sound
from gestures import BUILTIN_GESTURES, RAISED_FIST, SEAL, TWO_HAND_GESTURES, WOW
from jutsu import HoldTrigger, raised_fist, seal_point, wow_pose
from overlay import POSITIONS, OverlayPlayer, visual_clip
from tracking import palm_center

log = logging.getLogger(__name__)
ORB_SOUNDS = ("activation", "charge_loop", "blast", "power_down")  # the sounds an orb jurus may replace
MEAN_LENGTH = {1: 210, 2: 421}  # poses: one hand's pairwise distances; two hands: both, plus the wrist gap
TRY_SECONDS = 2.0   # how long the editor's Try button shows a visual that would otherwise be held


@dataclass(eq=False)
class Jurus:
    cfg: dict
    kind: str | None           # the visual's type: image | video | text | orb | vanish
    sound: str | None          # key into the sound set
    overlay: OverlayPlayer | None
    problem: str | None        # "gesture" | "sound" | "visual"
    hands: int                 # hands the gesture needs: 1 or 2
    trigger: HoldTrigger
    fired_at: float | None = None   # when it last fired (an exclusive jurus blocks others from here)
    holding: bool = False           # fired and its gesture is still being held
    seen_at: float = 0.0            # while holding: when the gesture was last seen

    @property
    def name(self) -> str:
        return self.cfg.get("name", "")

    @property
    def gesture(self) -> str:
        return str(self.cfg.get("gesture", ""))

    @property
    def live(self) -> bool:
        """Able to fire: enabled and nothing wrong with it."""
        return self.problem is None and self.cfg.get("enabled", True)

    @property
    def exclusive(self) -> bool:
        return bool(self.cfg.get("exclusive"))

    @property
    def looped(self) -> bool:
        return (self.cfg.get("sound") or {}).get("mode") == "loop"

    @property
    def volume(self) -> float:
        return _number((self.cfg.get("sound") or {}).get("volume"), 1.0)


def _number(value, default):
    """A hand-edited jurus.yaml may hold anything; a non-number must not reach the audio thread, the clock
    or the drawing code."""
    try:
        return value if isinstance(value, (int, float)) and math.isfinite(value) else default
    except OverflowError:  # an int too big for a float
        return default


def valid_templates(templates: list) -> list[dict]:
    """The recorded gestures that can be matched. A hand-edited gestures.json may hold anything, and
    matching runs every frame, so one bad entry would stop the app."""
    def ok(t) -> bool:
        return (isinstance(t, dict) and isinstance(t.get("name"), str) and t.get("hands") in (1, 2)
                and isinstance(t.get("mean"), list) and len(t["mean"]) == MEAN_LENGTH[t["hands"]]
                and all(_number(x, None) is not None for x in t["mean"])
                and all(_number(t.get(k), 0) > 0 for k in ("radius", "sensitivity")))
    return [t for t in templates if ok(t)]


def _clean_visual(visual: dict) -> dict:
    """The visual with the numbers and the position the drawing code relies on made safe; the size is kept to
    the editor's range."""
    position = visual.get("position")
    return {**visual, "size": min(max(_number(visual.get("size"), 0.4), 0.05), 3.0),
            "duration": _number(visual.get("duration"), "held"),
            "position": position if position in POSITIONS else "center"}


def _decode(ref: str, store, sr: int) -> np.ndarray:
    path = store.resolve(ref)
    if path is None:
        raise ValueError(f"not a media file: {ref!r}")
    buf = load_sound(path, sr)
    if not len(buf):
        raise ValueError(f"{ref} has no audio")
    return buf


def _sound_key(ref: str, store, sr: int, sounds: dict) -> str:
    """Name of `ref` in the sound set. A built-in is already in the audio engine; a media file is
    decoded into `sounds` once, however many jurus use it."""
    if ref.startswith("builtin:"):
        name = ref[len("builtin:"):]
        if name not in BUILTIN_SOUNDS:
            raise ValueError(f"no built-in sound {name!r}")
        return name
    if ref not in sounds:
        sounds[ref] = _decode(ref, store, sr)
    return ref


def build(jurus_cfg: list[dict], store, templates: list[dict], sr: int) -> tuple[list[Jurus], dict[str, np.ndarray], set[str]]:
    """The jurus, the sound set to hand to the audio engine, and the names of its looping sounds."""
    by_name = {t["name"]: t for t in valid_templates(templates)}
    sounds: dict[str, np.ndarray] = {}
    loops: set[str] = set()
    out = []
    for cfg in jurus_cfg:
        cfg = cfg if isinstance(cfg, dict) else {}  # keeps the output in step with the input; shows as a gesture problem
        visual = cfg.get("visual")
        # a tracker gap under 0.3 s isn't a release
        j = Jurus(cfg, visual.get("type") if isinstance(visual, dict) else None, None, None, None, 1,
                  HoldTrigger(_number(cfg.get("hold"), 0.3), gap=0.3))
        gesture = j.gesture
        template = by_name.get(gesture[len("rec:"):]) if gesture.startswith("rec:") else None
        if gesture in TWO_HAND_GESTURES or (template and template["hands"] == 2):
            j.hands = 2

        def check_gesture():
            if not (template is not None if gesture.startswith("rec:") else gesture in BUILTIN_GESTURES):
                raise ValueError(f"unknown gesture {gesture!r}")

        def load_sounds():
            if j.kind == "orb":  # the engine plays these; they replace the built-in orb sounds
                for key in ORB_SOUNDS:
                    if ref := (cfg.get("orb_sounds") or {}).get(key):
                        sounds[key] = _decode(ref, store, sr)
            elif sound := cfg.get("sound"):
                j.sound = _sound_key(sound["file"], store, sr, sounds)
                if j.looped:
                    loops.add(j.sound)

        def make_overlay():
            # the engine owns the orb and the vanish smoke (it resolves the smoke file itself)
            if visual and j.kind not in ("orb", "vanish"):
                clean = _clean_visual(visual)
                j.overlay = OverlayPlayer(visual_clip(clean, store.resolve(clean.get("file") or "")), clean)

        for problem, stage in (("gesture", check_gesture), ("sound", load_sounds), ("visual", make_overlay)):
            try:
                stage()
            except Exception as e:  # a bad file or setting must not stop the app; the UI shows the flag
                j.problem = problem
                log.warning("Jurus %r: %s unusable (%s)", j.name, problem, e)
                break
        out.append(j)
    return out, sounds, loops


def active_ids(gesture: str, hands, primary, lm, frame_h: int, cfg: dict, templates: list[dict]) -> set[str]:
    """Every gesture id true this frame: the classifier's (a held-up fist counts as RAISED_FIST), the
    two-hand poses, and recorded gestures. A one-hand recording is more specific than the built-in
    classification of that hand, so it replaces it; a two-hand one is added."""
    g = cfg["gesture"]
    ids = {RAISED_FIST if raised_fist(gesture, lm, frame_h, g["raise_above"]) else gesture}
    if seal_point(hands, g["finger_straight"]) is not None:
        ids.add(SEAL)
    if wow_pose(hands, g["finger_straight"]):
        ids.add(WOW)
    templates = valid_templates(templates)
    for n in (1, 2):  # matched per hand count, so a one-hand match can't hide a two-hand one
        if name := poses.best_match([t for t in templates if t["hands"] == n], hands, primary):
            if n == 1:
                ids -= {gesture, RAISED_FIST}
            ids.add(f"rec:{name}")
    return ids


def hands_center(hands) -> np.ndarray | None:
    """Mean of the palm centres, where the vanish smoke lands."""
    return np.mean([palm_center(h.lm) for h in hands], axis=0) if hands else None


class Rules:
    def __init__(self, jurus: list[Jurus], audio):
        self.jurus, self.audio = jurus, audio
        self._voice: Jurus | None = None   # whose one-shot sound was requested last: the one that is playing

    @property
    def orb_jurus(self) -> Jurus | None:
        """The orb jurus, if there is one that can run; the engine drives it."""
        return next((j for j in self.jurus if j.kind == "orb" and j.live), None)

    def _exclusive_busy(self, me: Jurus, t: float) -> bool:
        return any(j is not me and j.exclusive and j.fired_at is not None
                   and ((j.sound and t < j.fired_at + self.audio.duration(j.sound)) or (j.overlay and j.overlay.busy))
                   for j in self.jurus)

    def _allowed(self, j: Jurus, t: float, two_hands: bool, blocked: bool) -> bool:
        return j.live and not blocked and not (j.hands == 1 and two_hands) and not self._exclusive_busy(j, t)

    def _play(self, j: Jurus, loop: bool):
        if j.sound:
            if loop:
                self.audio.set_loop(j.sound, j.volume)
            else:
                self.audio.play(j.sound, j.volume, j.exclusive)
                self._voice = j

    def _release(self, j: Jurus, t: float):
        j.holding = False
        if j.sound:
            if j.looped:
                self.audio.set_loop(j.sound, 0.0)
            elif not j.exclusive and self._voice is j:  # else a later sound already cut it (maybe under the same name)
                self.audio.stop(j.sound)
        if j.overlay:
            j.overlay.release(t)

    def release_all(self, t: float):
        """Let go of everything held: these jurus are being replaced, and the new ones never fired, so nothing
        else would stop a held loop."""
        for j in self.jurus:
            if j.holding:
                self._release(j, t)

    def update(self, active: set[str], t: float, two_hands: bool, blocked: bool) -> list[Jurus]:
        """Advance one frame; returns the jurus that fired. `blocked`: the orb or a vanish is busy.
        Orb jurus are skipped: the engine drives the orb."""
        for j in self.jurus:
            if j.overlay:
                j.overlay.update(t)
        todo = [j for j in self.jurus if j.kind != "orb"]
        for j in todo:  # let go first, so a stop never lands after the play that replaces it
            if not j.holding:
                continue
            if j.gesture in active:
                j.seen_at = t
            # a dropout shorter than the trigger's gap isn't letting go (it won't re-arm either); the
            # engine passes blocked while a vanish runs, which must not cut the vanish jurus's own sound
            if (t - j.seen_at > j.trigger.gap
                    or not self._allowed(j, t, two_hands, blocked and j.kind != "vanish")):
                self._release(j, t)
        fired = []
        for j in todo:  # in list order: the first of two simultaneous exclusive jurus wins
            if j.trigger.update(j.gesture in active, t, self._allowed(j, t, two_hands, blocked)):
                j.fired_at, j.seen_at, j.holding = t, t, True
                self._play(j, j.looped)
                if j.overlay:
                    j.overlay.start(t)
                fired.append(j)
        return fired

    def try_fire(self, name: str, t: float) -> Jurus | None:
        """The editor's Try button: fire a jurus now, whatever the gesture or blocking. The sound plays
        once even in loop mode and a held visual shows for TRY_SECONDS. None for an unknown name or a
        jurus with a problem."""
        j = next((j for j in self.jurus if j.name == name), None)
        if j is None or j.problem:
            return None
        j.fired_at = t   # counts as a fire for exclusive blocking; not `holding`, so the next frame doesn't stop it
        self._play(j, loop=False)
        if j.overlay:
            held = not isinstance(j.overlay.visual.get("duration"), (int, float))
            j.overlay.start(t, TRY_SECONDS if held else None)
        return j

    def draw(self, frame: np.ndarray, t: float, palm: tuple[float, float, float] | None, preview: bool):
        for j in self.jurus:
            if j.overlay:
                j.overlay.draw(frame, t, palm, preview)
