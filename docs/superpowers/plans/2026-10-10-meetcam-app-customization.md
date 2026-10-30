# MeetCam App + Customization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One Qt window (preview + Jurus / Gesture / Settings tabs) where every gesture → sound → visual
pairing is editable, including user-recorded gestures, with all current effects kept.

**Architecture:** Today's per-frame loop moves from `main.py` into `engine.py` (a `QThread`). Hard-wired
features (finger SFX, seal, wow, raised fist) become data: `jurus.yaml` entries run by `rules.py`, which uses
`poses.py` (recorded gestures), `overlay.py` (image/video/text) and the existing orb/vanish code.
`store.py` owns the per-user data folder; `ui.py` edits it and hands rebuilt rules to the engine.

**Tech Stack:** Python 3.12 (`.venv`), PySide6 (new), MediaPipe, OpenCV, NumPy, Pillow, sounddevice,
soundfile, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-meetcam-app-customization-design.md`

## Global Constraints

- Branch `app-customization`; one commit per task; PR to `main` at the end with `"/c/Program Files/GitHub CLI/gh.exe"`.
- Test command (Git Bash, repo root): `.venv/Scripts/python -m pytest -q`. All tests pass at the end of every task.
- Only new dependency: `PySide6`. Add `Pillow` to requirements.txt (already installed via matplotlib).
- Never commit media: `assets/audio/*`, `assets/video/*`, `assets/background.png` stay gitignored.
- Defaults ship no copyrighted media: only `builtin:` sounds, the procedural orb and text.
- Effect videos: one-colour backdrop keyed by `jutsu.load_clip`, otherwise shown as-is. No AI segmentation.
- Positions are "as seen in the mirrored preview"; the meeting gets the unmirrored frame.
- Every user-visible string goes through `i18n.t(key)` and exists in both `id` and `en`.
- Kept behaviour: one sound at a time; non-exclusive sounds stop on release; exclusive sounds play to the
  end and block other jurus until their sound and visual finish; `gesture.two_hand_grace` blocks
  one-hand jurus; orb charging/blast/cooldown blocks other jurus; fire once per hold, re-arm after
  release, `gap` 0.3 s tolerated; max one `orb` and one `vanish` jurus.
- Tests that touch Qt run offscreen: `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` before any Qt import.

## Review Focus

1. Media file with spaces/non-ASCII name (e.g. `stiker é.png`): imports and loads (OpenCV can't open
   non-ASCII paths on Windows) → Task 1 sanitises names, Task 3 loads images via Pillow; tests in both.
2. Jurus pointing at a recorded gesture that was deleted: jurus shows ⚠ "gesture", never fires, no crash → Task 6 test.
3. Recording where the hand is lost or the second hand flickers: clear error, no garbage template → Task 2 test.
4. Corrupt or unsupported sound/image/video file: that jurus ⚠, others still fire → Task 6 test.
5. Camera index that no longer exists (webcam unplugged): message in the preview area, app keeps running → Task 7 test.

---

### Task 1: `store.py`, the per-user data folder

**Files:**
- Create: `store.py`, `tests/conftest.py`, `tests/test_store.py`

**Interfaces:**
- Produces:
  - `APP_NAME = "MeetCam"`
  - `DEFAULT_SETTINGS = {"camera_index": 0, "mic": None, "speaker": None, "meeting_output": "CABLE Input", "virtual_camera": True, "language": None, "mirror_preview": True, "skeleton": False}`
  - `DEFAULT_JURUS: list[dict]`, the spec's first-run defaults:
    - Bola Energi: `OPEN_PALM`, visual `{type: orb}`.
    - Beep 1–4: `ONE`…`FOUR`, sound `builtin:finger_1`…`builtin:finger_4`. FOUR uses mode `loop`, the others `once`. hold 0.3.
    - Menghilang: `SEAL`, exclusive, sound `builtin:finger_3`, visual `{type: vanish}`.
    - WOW: `WOW`, exclusive, sound `builtin:finger_2`, visual `{type: text, text: "WOW!", color: "#ffd400", position: top, size: 0.25, duration: 2.0}`.
    - Every entry has `enabled: true` and `exclusive` set (false unless stated).
  - `default_root() -> Path`: `QStandardPaths.writableLocation(AppDataLocation)` after `QCoreApplication.setApplicationName(APP_NAME)`. Import Qt inside the function.
  - `class Store(root: Path)`. It creates `root/media`. Members:
    - Attributes: `.root`, `.background` (`root/"background.png"`), `.problems: list[str]`.
    - `load_settings() -> dict` returns `DEFAULT_SETTINGS` overlaid with the file. `save_settings(d)`.
    - `load_jurus() -> list[dict]` returns a copy of `DEFAULT_JURUS` if the file is missing. `save_jurus(lst)`.
    - `load_gestures() -> list[dict]` returns `[]` if the file is missing. `save_gestures(lst)`.
    - `import_media(src: Path) -> str` returns `"media/<safe name>"`.
    - `resolve(ref: str) -> Path | None`: `"media/x"` → `root/"media"/"x"`; `"builtin:..."` or empty → `None`.

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`: `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`.

```python
def test_missing_files_give_defaults(tmp_path):
    s = Store(tmp_path)
    assert s.load_settings() == DEFAULT_SETTINGS and s.load_jurus() == DEFAULT_JURUS and s.load_gestures() == []

def test_round_trip(tmp_path):
    s = Store(tmp_path)
    s.save_settings({**DEFAULT_SETTINGS, "language": "en"}); s.save_jurus(DEFAULT_JURUS[:1]); s.save_gestures([{"name": "ok"}])
    s2 = Store(tmp_path)
    assert s2.load_settings()["language"] == "en" and s2.load_jurus() == DEFAULT_JURUS[:1] and s2.load_gestures() == [{"name": "ok"}]

def test_broken_file_is_set_aside_and_defaults_used(tmp_path):
    (tmp_path / "jurus.yaml").write_text("- [unclosed", encoding="utf-8")
    s = Store(tmp_path)
    assert s.load_jurus() == DEFAULT_JURUS
    assert (tmp_path / "jurus.yaml.broken").exists() and s.problems

def test_import_media_copies_sanitises_and_dedupes(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    a = src / "stiker é.png"; a.write_bytes(b"A")
    s = Store(tmp_path / "data")
    ref = s.import_media(a)
    assert ref == "media/stiker_.png" and s.resolve(ref).read_bytes() == b"A"
    assert s.import_media(a) == ref                      # same content → same file
    b = src / "stiker é.png"; b.write_bytes(b"B")
    assert s.import_media(b) == "media/stiker__2.png"   # same name, different content
    assert s.resolve("builtin:finger_1") is None
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_store.py -q`. Expected: FAIL with `ModuleNotFoundError: store`.

- [ ] **Step 3: Implement `store.py`**

- Files are `settings.yaml`, `jurus.yaml` and `gestures.json`.
- Writes are atomic: write `<file>.tmp`, then `os.replace`.
- When parsing fails: `os.replace(file, file + ".broken")`, append `"<file name>"` to `problems`, and return the default.
- Safe name: `re.sub(r"[^A-Za-z0-9._-]+", "_", name)`.
- On a clash with different bytes, add a suffix: `_2`, `_3`, and so on.

- [ ] **Step 4: Run** the same command. Expected: 4 passed. Then run the full suite: all pass.

- [ ] **Step 5: Commit** `git add store.py tests/conftest.py tests/test_store.py && git commit -m "Add per-user data store with safe saves and media import"`

---

### Task 2: `poses.py`, recorded gestures

**Files:**
- Create: `poses.py`, `tests/test_poses.py`

**Interfaces:**
- Consumes:
  - `tracking.Hand(lm, world)`, `tracking.palm_size`, `tracking.WRIST`.
  - Test helpers: `test_logic.hand3d`, `test_logic.pick`.
- Produces:
  - Constants: `RADIUS_FLOOR = 0.05`, `DEFAULT_SENSITIVITY = 1.5`.
  - `hand_descriptor(world) -> np.ndarray`, shape (210,): pairwise distances ÷ `palm_size(world)`.
  - `make_template(name: str, frames: list[list[Hand]]) -> dict`. Returns `{"name", "hands": 1|2, "mean": list[float], "radius": float, "sensitivity": 1.5}`.
    - Raises `ValueError("no_hand")` when no frame has a hand.
    - Raises `ValueError("unsteady")` when under 50% of frames share the most common hand count.
  - `distance(template, hands: list[Hand], primary: Hand | None) -> float | None`.
    - 1-hand templates use `primary`. 2-hand templates need `len(hands) == 2`.
    - Returns `None` on a count mismatch.
  - `score(template, hands, primary) -> float | None`: `1 - d / (radius * sensitivity)`.
  - `best_match(templates, hands, primary) -> str | None`: the name of the nearest template with score > 0.

- [ ] **Step 1: Write the failing tests**

```python
def rotate(world, rvec):  # cv2.Rodrigues rotation about the wrist
def frames_of(world, n=30, noise=0.0015): ...  # n single-hand frames, Gaussian jitter in metres, lm = world*1000+offset

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
    a = Hand(*hand3d(pick(2), False)); lm4, w4 = hand3d(pick(4), True)
    b = Hand(lm4 + np.array([150, 0, 0], np.float32), w4)
    t = make_template("duo", [[a, b]] * 30)
    assert best_match([t], [b, a], b) == "duo" and distance(t, [a], a) is None

def test_recording_without_a_steady_hand_is_refused():
    with pytest.raises(ValueError, match="no_hand"): make_template("x", [[]] * 30)
    h = frames_of(hand3d()[1], n=1)[0][0]
    with pytest.raises(ValueError, match="unsteady"): make_template("x", [[h]] * 10 + [[h, h]] * 10 + [[]] * 10)
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_poses.py -q`. Expected: FAIL with `ModuleNotFoundError: poses`.

- [ ] **Step 3: Implement `poses.py`**

- Distances: `np.linalg.norm(w[:, None] - w[None], axis=2)[np.triu_indices(21, 1)]`.
- 2-hand descriptor: sort the hands by `lm[WRIST, 0]`. Concatenate both hand descriptors plus one gap term: the wrist pixel distance ÷ mean pixel `palm_size`.
- `distance` is the RMS of the difference. For 2 hands, take the minimum over both orders.
- `radius = max(RADIUS_FLOOR, 95th percentile of the samples' RMS distance to the mean)`.
- Add `# ponytail: shape only, not direction (thumbs-up == thumbs-down); add the palm normal to the descriptor if users ask`.

- [ ] **Step 4: Run** the same command. Expected: 5 passed. Then run the full suite: all pass.

- [ ] **Step 5: Commit** `git add poses.py tests/test_poses.py && git commit -m "Add recorded-gesture templates and matching"`

---

### Task 3: `overlay.py`, image / video / text visuals

**Files:**
- Create: `overlay.py`, `tests/test_overlay.py`
- Modify: `requirements.txt` (add `PySide6`, `Pillow`). Then run `.venv/Scripts/python -m pip install PySide6`.

**Interfaces:**
- Consumes:
  - `jutsu.Clip`, `jutsu.load_clip(path)`, `jutsu.draw_clip(frame, clip, i, x0, y0, scale, gain)`.
- Produces:
  - Constants:
    - `POSITIONS = ("hand", "full", "top_left", "top", "top_right", "left", "center", "right", "bottom_left", "bottom", "bottom_right")`
    - `MARGIN = 0.04` (fraction of frame height)
    - `RELEASE_FADE = 0.2` (moved here from jutsu)
  - Loaders, each returning a `Clip`:
    - `image_clip(path: Path) -> Clip`: Pillow, RGBA. A GIF yields all its frames at fps = 1000 / mean frame duration in ms. `origin` is (0, 0), `source_width` is the image width, `peak` is 0 and `anchor` is the centre.
    - `text_clip(text: str, color: str, px: int = 160) -> Clip`: QPainter onto a `QImage.Format_ARGB32`, with a dark outline. Needs a `QGuiApplication`. The test creates one when `QGuiApplication.instance()` is `None`.
  - `visual_clip(visual: dict, path: Path | None) -> Clip` dispatches image/video/text and raises on failure.
  - `place(clip, position, size, frame_w, frame_h, palm: tuple[float, float, float] | None, preview: bool) -> tuple[int, int, float] | None` returns `(x0, y0, scale)`.
    - `palm` = (x, y, palm_size) in output (unmirrored) pixels.
    - Returns `None` for `hand` without a palm.
  - `class OverlayPlayer(clip: Clip, visual: dict)`:
    - `busy` property.
    - `start(t, seconds: float | None = None)`: `seconds` overrides `held`, for the Try button.
    - `release(t)`, `update(t)`, `draw(frame, t, palm, preview: bool)`.

- [ ] **Step 1: Write the failing tests**

```python
def solid(w, h): return Clip([(np.full((h, w, 3), 200, np.uint8), np.full((h, w), 255, np.uint8))], 30.0, 0, (w/2, h/2), (0, 0), w)

def test_grid_positions_are_as_seen_in_the_preview():
    c = solid(100, 100)                                     # size 0.25 of a 400-high frame → scale 1.0
    assert place(c, "top_left", 0.25, 800, 400, None, preview=True)[:2] == (16, 16)
    assert place(c, "top_left", 0.25, 800, 400, None, preview=False)[:2] == (800 - 16 - 100, 16)
    assert place(c, "bottom_right", 0.25, 800, 400, None, preview=True)[:2] == (800 - 16 - 100, 400 - 16 - 100)
    assert place(c, "center", 0.25, 800, 400, None, preview=True)[:2] == (350, 150)

def test_hand_position_follows_the_palm_mirrored_for_the_preview():
    x0, y0, s = place(solid(100, 100), "hand", 2.0, 800, 400, (200, 300, 50), preview=False)
    assert (x0, y0, s) == (150, 250, 1.0)
    assert place(solid(100, 100), "hand", 2.0, 800, 400, (200, 300, 50), preview=True)[:2] == (550, 250)
    assert place(solid(100, 100), "hand", 2.0, 800, 400, None, preview=True) is None

def test_full_keeps_the_clip_framing():
    c = Clip(solid(50, 40).frames, 30.0, 0, (25, 20), (10, 20), 200)   # cropped from a 200-wide source
    assert place(c, "full", 1.0, 800, 450, None, preview=False) == (40, 80, 4.0)

def test_gif_and_unicode_named_png_load_with_alpha(tmp_path): ...  # 3-frame GIF (Pillow save, duration=100) → 3 frames, fps 10;
                                                               # "é.png" RGBA with a transparent half → alpha has 0 and 255
def test_text_renders_a_visible_sprite(): c = text_clip("WOW!", "#ffd400"); assert c.frames[0][1].max() == 255 and c.frames[0][1].min() == 0

def test_player_held_fades_on_release_and_timed_ends_by_itself():
    p = OverlayPlayer(solid(10, 10), {"duration": "held"}); p.start(0.0); p.update(5.0); assert p.busy
    p.release(5.0); p.update(5.0 + RELEASE_FADE + 0.01); assert not p.busy
    q = OverlayPlayer(solid(10, 10), {"duration": 2.0}); q.start(0.0); q.update(1.9); assert q.busy; q.update(2.01); assert not q.busy
    r = OverlayPlayer(solid(10, 10), {"duration": "held"}); r.start(0.0, seconds=2.0); r.update(2.01); assert not r.busy
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_overlay.py -q`. Expected: FAIL with `ModuleNotFoundError: overlay`.

- [ ] **Step 3: Implement `overlay.py`**

Grid maths, in preview coordinates:
- Margin `m = round(MARGIN * frame_h)`.
- Content size is `w, h = crop size × scale`, with `scale = size * frame_h / crop_h`.
- Column x0: left `m`, middle `(W - w) // 2`, right `W - m - w`. Rows work the same way with H.
- For the output frame: `x0_out = W - x0_preview - w`.
- `hand`: `scale = size * palm_size / crop_h`, centred on the palm. In the preview, `x = W - x`.
- `full`: `scale = W / source_width`. Output: `x0 = origin_x * scale`, `y0 = origin_y * scale`. Preview: `x0 = W - (origin_x * scale + w)`.

Drawing:
- The frame index loops.
- After `release`, `gain = 1 - (t - released) / RELEASE_FADE`.
- Draw with `draw_clip`.

- [ ] **Step 4: Run** the same command. Expected: 6 passed. Then run the full suite: all pass.

- [ ] **Step 5: Commit** `git add overlay.py tests/test_overlay.py requirements.txt && git commit -m "Add image, video and text overlays with preview-relative placement"`

---

### Task 4: `audio.py`, runtime sound sets

**Files:**
- Modify: `audio.py`. Remove `find_asset`, `ASSET_DIR`, `update_fingers`, `play_pose`, `self.fingers` and `self.poses`. Change `AudioEngine.__init__` and `Mixer`.
- Modify: `tests/test_logic.py`:
  - Delete the finger-SFX and pose-sound tests. Task 6 re-creates them as rules tests: lines ~323–416 and ~690–740.
  - Keep the Mixer, Fifo, meeting and loading tests.
  - Delete `test_asset_lookup_accepts_any_supported_extension`.

**Interfaces:**
- Produces:
  - `BUILTIN_SOUNDS = ("activation", "charge_loop", "blast", "power_down", "finger_1", "finger_2", "finger_3", "finger_4")`
  - `class AudioEngine(cfg: dict, meeting_cfg: dict | None = None)`:
    - `set_sounds(sounds: dict[str, np.ndarray], loops: set[str])`: merged over the synthesized builtins. `"charge_loop"` is always a loop.
    - `play(name, gain=1.0, exclusive=False)`, `stop(name)`, `set_loop(name, gain, rate=1.0)`. Each one fans out to every mixer.
    - `duration(name) -> float`, in seconds.
    - Unchanged: `update(prev, state, charge, blast_power, silent_fade)`, `meeting_on`, `muted`, `close`.
  - `Mixer.set_sounds(sounds, loops)`: swaps `self.sounds` and `self._loops` by reference assignment. Voices already playing keep their buffers.
  - `list_devices(host_api: str, kind: str) -> list[str]`: device names on that host API with `max_<kind>_channels > 0`.
  - Meeting mic guard: if the resolved mic name contains "CABLE", `_open_meeting` raises `LookupError("mic is the virtual cable")`. Meeting audio then goes OFF, which avoids a feedback loop.

- [ ] **Step 1: Write the failing tests** (in `tests/test_logic.py`, next to the Mixer tests)

```python
def test_sound_set_can_be_swapped_at_runtime():
    a = AudioEngine({**M_CFG["audio"], "enabled": False})
    a.set_sounds({"ding": np.ones(4800, np.float32)}, set())
    assert a.duration("ding") == pytest.approx(0.1) and "finger_1" in a.mixer.sounds
    a.play("ding"); assert np.abs(a.mixer.render(256)).max() > 0

def test_loop_sounds_follow_the_set():
    a = AudioEngine({**M_CFG["audio"], "enabled": False})
    a.set_sounds({"hum": np.ones(4800, np.float32)}, {"hum"})
    a.set_loop("hum", 1.0); assert np.abs(a.mixer.render(4096)).max() > 0
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_logic.py -q -k "swapped or follow_the_set"`. Expected: FAIL with `TypeError` (old constructor signature).

- [ ] **Step 3: Implement the interface above in `audio.py`.** Remove the `finger_sfx` section from `config.yaml` in the same commit.

- [ ] **Step 4: Run** the full suite. Expected: all pass. (`main.py` can't run from here until Task 7 replaces it. Its import still works, so `test_virtual_camera_*` keep passing.)

- [ ] **Step 5: Commit** `git commit -am "Audio engine takes its sound set at runtime"`

---

### Task 5: generalise the orb trigger and the vanish; drop `PoseClip`

**Files:**
- Modify: `gestures.py` (`OrbMachine`, gesture ids), `jutsu.py` (`Vanish`; delete `PoseClip`, `RELEASE_FADE`), `config.yaml` (delete the `vanish`, `wow` and raised-fist sections; add `vanish_defaults: {capture_delay: 5, smoke_height: 1.8}`), `tests/test_logic.py`

**Interfaces:**
- Produces:
  - In `gestures.py`:
    - `RAISED_FIST, SEAL, WOW = "RAISED_FIST", "SEAL", "WOW"`
    - `BUILTIN_GESTURES = (ONE, TWO, THREE, FOUR, OPEN_PALM, FIST, RAISED_FIST, SEAL, WOW)`
    - `TWO_HAND_GESTURES = {SEAL, WOW}`
  - `OrbMachine(cfg, trigger: str = OPEN_PALM)`: `self.trigger` replaces OPEN_PALM everywhere. The FOUR-while-charging tolerance applies only when `trigger == OPEN_PALM`.
  - `Vanish(cfg, frame_shape)`, where `cfg = {"smoke": str | None, "background": str, "capture_delay": float, "smoke_height": float}`:
    - `fire(at: np.ndarray, t) -> None` toggles, or explains that the room photo is missing.
    - `update(t) -> None` advances the mix.
    - `busy`, `mix`, `message`, `start_capture`, `capture_tick`, `apply_plate`, `draw_smoke` are unchanged.
    - `HoldTrigger` moves out of `Vanish`; the rules own triggering.

- [ ] **Step 1: Write the failing tests**

```python
def test_orb_can_be_summoned_by_another_gesture():
    m = OrbMachine(M_CFG, trigger=TWO); m.update(TWO, AIM, 0.0); m.update(TWO, AIM, 0.1)
    assert m.state == ACTIVE
    m2 = OrbMachine(M_CFG, trigger=TWO); m2.update(OPEN_PALM, AIM, 0.0)
    assert m2.state == HIDDEN

def test_four_tolerance_only_for_open_palm_trigger(): ...  # trigger=TWO, charging, then FOUR → FADING
```

Adapt `test_without_a_background_the_seal_explains_instead_of_vanishing` and `test_capture_then_seal_vanishes_and_seal_again_returns` to `fire()` + `update()`. Delete the `PoseClip` tests; `OverlayPlayer` covers them.

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_logic.py -q`. Expected: the new tests FAIL (`TypeError: unexpected keyword 'trigger'`).

- [ ] **Step 3: Implement the interface above.**

- [ ] **Step 4: Run** the full suite. Expected: all pass.

- [ ] **Step 5: Commit** `git commit -am "Orb trigger and vanish are driven from outside; drop PoseClip"`

---

### Task 6: `rules.py`, jurus runtime

**Files:**
- Create: `rules.py`, `tests/test_rules.py`

**Interfaces:**
- Consumes:
  - `Store.resolve` (Task 1).
  - `poses.best_match` (Task 2).
  - `overlay.visual_clip`, `OverlayPlayer` (Task 3).
  - `audio.load_sound`, `AudioEngine.play/stop/set_loop/duration` (Task 4).
  - `gestures.*` ids (Task 5).
  - `jutsu.HoldTrigger`, `seal_point`, `wow_pose`, `raised_fist`.
- Produces:
  - `@dataclass Jurus` with fields:
    - `cfg: dict`
    - `kind: str | None` (`image`, `video`, `text`, `orb`, `vanish` or `None`)
    - `sound: str | None` (key into the sound set)
    - `overlay: OverlayPlayer | None`
    - `problem: str | None` (`"sound"`, `"visual"`, `"gesture"` or `None`)
    - `hands: int`
    - `trigger: HoldTrigger`
  - `build(jurus_cfg: list[dict], store, templates: list[dict], sr: int) -> tuple[list[Jurus], dict[str, np.ndarray], set[str]]`. Returns the jurus, the sound set and the loop names.
    - Sound keys: `builtin:x` → `x`; `media/...` → the ref itself.
    - Orb: `orb_sounds` are loaded under `activation`, `charge_loop`, `blast` and `power_down`.
    - Failures set `problem`. A jurus with a problem or `enabled: false` never fires.
  - `active_ids(gesture: str, hands, primary, lm, frame_h: int, cfg: dict, templates) -> set[str]`:
    - The base is `{gesture}`.
    - `RAISED_FIST` replaces FIST when `raised_fist(...)` holds, using `cfg["gesture"]["raise_above"] = 0.6`. Move that key into `config.yaml`'s gesture section.
    - Adds `SEAL` when `seal_point` is not None, and `WOW` when `wow_pose` holds.
    - A recorded match `rec:<name>` replaces the built-in id for a 1-hand template, and is added for a 2-hand one.
  - `hands_center(hands) -> np.ndarray | None`: the mean of the palm centres.
  - `class Rules(jurus: list[Jurus], audio)`:
    - `update(active: set[str], t, two_hands: bool, blocked: bool) -> list[Jurus]` returns the jurus that fired this frame. Orb jurus are skipped; the engine drives them.
    - `try_fire(name, t) -> Jurus | None`.
    - `draw(frame, t, palm, preview: bool)`.
    - `orb_jurus -> Jurus | None`.

- [ ] **Step 1: Write the failing tests** (`FakeAudio` records `("play", name, gain, exclusive)`, `("stop", name)` and `("loop", name, gain)`; `duration()` returns 1.0)

```python
def J(name, gesture, **kw) -> dict: ...  # jurus dict: builtin:finger_1 sound, hold 0.3, kw overrides
def rig(*cfgs) -> tuple[FakeAudio, Rules]: ...  # build(list(cfgs), Store(tmp), [], 48000) → Rules(jurus, FakeAudio())

def test_fires_once_per_hold_and_rearms_after_release():
    a, r = rig(J("one", ONE))
    fired = [r.update({ONE}, t, False, False) for t in np.arange(0, 1.0, 0.04)]
    assert sum(map(bool, fired)) == 1
    r.update(set(), 1.5, False, False)
    assert [x for t in np.arange(1.6, 2.2, 0.04) for x in r.update({ONE}, t, False, False)]

def test_release_stops_a_non_exclusive_sound_and_new_jurus_cuts_old(): ...  # ("stop", key) after release
def test_exclusive_blocks_others_until_its_sound_ends():
    # WOW (exclusive, duration 1.0) fires at 0.3; ONE held from 0.4 → no fire before 1.3, fires after
def test_two_hand_grace_blocks_one_hand_jurus_but_not_two_hand(): ...
def test_blocked_stops_an_already_firing_sound(): ...  # orb busy → ("stop", key)
def test_loop_mode_plays_while_held(): ...  # ("loop", key, vol) while held, ("loop", key, 0.0) after
def test_vanish_jurus_is_returned_when_fired(): ...
def test_problems_are_flagged_and_never_fire(tmp_path):
    # rec:gone (no template) → "gesture"; media/bad.mp3 with junk bytes → "sound"; media/bad.png junk → "visual"
    # a healthy jurus in the same list still fires
def test_recorded_match_replaces_the_finger_count(): ...  # template recorded from hand3d(pick(2)) → active_ids has rec:peace, not TWO
def test_raised_fist_replaces_fist(): ...
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_rules.py -q`. Expected: FAIL with `ModuleNotFoundError: rules`.

- [ ] **Step 3: Implement `rules.py`**

Behaviour rules (Global Constraints), applied per jurus per frame:
- `allowed = not problem and enabled and not blocked and not (hands == 1 and two_hands) and not other_exclusive_busy`.
- `other_exclusive_busy`: another exclusive jurus whose `t < fired_at + audio.duration(sound)`, or whose overlay is busy.
- On fire:
  - once: `play(sound, vol, exclusive)`.
  - loop: `set_loop(sound, vol)`.
  - `overlay.start(t)`.
- When the gesture leaves `active`, or `allowed` drops while the jurus is firing:
  - a non-exclusive once-sound gets `stop`;
  - a loop gets `set_loop(sound, 0)`;
  - the overlay gets `release`.

- [ ] **Step 4: Run** the same command. Expected: 10 passed. Then run the full suite: all pass.

- [ ] **Step 5: Commit** `git add rules.py tests/test_rules.py config.yaml && git commit -m "Add jurus rules: triggers, exclusivity, recorded gestures"`

---

### Task 7: `engine.py`, the frame loop in a QThread

**Files:**
- Create: `engine.py`, `tests/test_engine.py`
- Modify: `main.py`. It becomes `QApplication` + `Store(default_root())` + `ui.MainWindow`. `open_camera`, `open_virtual_camera`, `draw_skeleton` and `facing` move to `engine.py`. Move `test_virtual_camera_*` to `tests/test_engine.py`.

**Interfaces:**
- Consumes:
  - `rules.build/Rules/active_ids/hands_center`, `Store`, `AudioEngine`, `OrbMachine(trigger=…)`, `Vanish`.
  - `EnergyOrb`: needs `cfg["orb"]["style"]` / `["video"]` from the orb jurus `visual.file`.
- Produces:
  - `class Engine(QThread)`, constructed as `Engine(cfg: dict, store: Store, settings: dict, tracker=None, audio=None, capture=None)`. Arguments left as `None` are opened in `run()`.
  - Signals:
    - `frame_ready = Signal(QImage)`
    - `status = Signal(dict)`, with keys `fps`, `vcam`, `meeting`, `gesture`, `message`, `scores: dict[str, float | None]`, `invisible`
    - `recorded = Signal(object)`: a template dict, or an error key string `"no_hand"` / `"unsteady"`
  - `step(frame: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]` returns `(outgoing, preview)` and is pure per frame:
    1. Scene (orb, vanish).
    2. `preview = flip(scene.copy())` if mirroring.
    3. `rules.draw(outgoing, preview=False)` and `rules.draw(preview, preview=True)`.
  - `reload(jurus, sounds, loops, templates)`: thread-safe. It stores a tuple that `step` swaps in at the start of the next frame and calls `audio.set_sounds`.
  - Commands, queued and run by the engine thread: `capture_room()`, `record()`, `try_jurus(name)`, `set_settings(settings)` (reopens camera/audio/vcam as needed), `stop()`.
  - Flags: `effects_on`, `skeleton`. `muted` maps to audio.
  - Camera backend: `cv2.CAP_DSHOW` on win32, else `cv2.CAP_ANY`.

- [ ] **Step 1: Write the failing tests**

```python
class FakeTracker:  detect = lambda self, f, t: []; close = lambda self: None

def make_engine(store): return Engine(M_CFG, store, DEFAULT_SETTINGS, tracker=FakeTracker(), audio=AudioEngine({**M_CFG["audio"], "enabled": False}))

def test_step_returns_unmirrored_output_and_mirrored_preview(tmp_path):
    e = make_engine(Store(tmp_path)); f = np.zeros((360, 640, 3), np.uint8); f[:, :10] = 255
    out, prev = e.step(f, 0.0)
    assert out[:, :10].min() == 255 and prev[:, -10:].min() == 255

def test_text_overlay_reads_the_same_in_preview_and_output(tmp_path):
    # jurus WOW via try_jurus → the preview region at "top" equals the un-flipped sprite, and so does the outgoing region at the mirrored x

def test_reload_swaps_jurus_between_frames(tmp_path): ...
def test_missing_camera_reports_instead_of_crashing(tmp_path):
    e = make_engine(Store(tmp_path)); e.settings = {**DEFAULT_SETTINGS, "camera_index": 97}
    msgs = []; e.status.connect(msgs.append)
    assert e._open_camera() is False and msgs[-1]["message"] == "camera_failed"
```

`make_engine`, `FakeTracker` and `M_CFG` (config.yaml loaded) live in `tests/conftest.py`, so the UI tests can reuse them.

`run()` = `_open_camera() -> bool`, then the loop. On failure it emits `status` with `message: "camera_failed"` and waits for `set_settings`.

Vanish cfg = `{"smoke": <resolved visual.file or None>, "background": str(store.background), **cfg["vanish_defaults"]}`.

Orb cfg = `cfg` with `orb.style = "video"` and `orb.video = <resolved file>` when the orb jurus has a file, else `"procedural"`.

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_engine.py -q`. Expected: FAIL with `ModuleNotFoundError: engine`.

- [ ] **Step 3: Implement `engine.py`** by moving `main.py`'s loop body into `step`, wired to Rules / orb jurus / vanish jurus as in the spec's Architecture section. Add a recording state:
- `record()` starts a 3 s countdown, shown through `message`.
- Then 2 s of collecting `[hands]` per frame.
- Then `make_template("", frames)` → emit `recorded`.

- [ ] **Step 4: Run** the full suite. Expected: all pass.

- [ ] **Step 5: Commit** `git add engine.py main.py tests && git commit -m "Move the frame loop into a QThread engine"`

---

### Task 8: `i18n.py` + main window + Settings tab

**Files:**
- Create: `i18n.py`, `ui.py`, `tests/test_ui.py`

**Interfaces:**
- Consumes:
  - `Engine` (Task 7), `Store` (Task 1), `audio.list_devices`.
  - `QMediaDevices.videoInputs()`: the descriptions; the index is the OpenCV index.
- Produces:
  - `i18n.STRINGS = {"id": {...}, "en": {...}}`, `set_language(code: str | None)`, `t(key: str, **fmt) -> str`. `None` follows `QLocale.system().name()`: `"id"` if it starts with `id`, else `"en"`.
  - `ui.MainWindow(store: Store, cfg: dict, engine: Engine | None = None)` (tests pass an unstarted engine). Contents:
    - The preview `QLabel`.
    - Status lights: virtual camera, meeting audio, FPS, each with a tooltip saying what to install when OFF.
    - Buttons: Effects, Mute, Room photo, Skeleton.
    - `QShortcut`s for Space/M/B/D/Q.
    - `tabs` with `settings_tab`.
  - `ui.SettingsTab(store, engine)`: camera, mic, speaker, meeting output, virtual camera, language, mirror. Every change triggers `store.save_settings`, then `engine.set_settings`.

- [ ] **Step 1: Write the failing tests**

```python
def test_every_string_exists_in_both_languages():
    assert STRINGS["id"].keys() == STRINGS["en"].keys()

def test_main_window_builds_and_settings_save(tmp_path, qapp):   # qapp: QApplication fixture in conftest.py
    w = MainWindow(Store(tmp_path), M_CFG, make_engine(Store(tmp_path)))
    w.settings_tab.mirror.setChecked(False)
    assert Store(tmp_path).load_settings()["mirror_preview"] is False
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_ui.py -q`. Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `i18n.py`, `MainWindow`, `SettingsTab`; add the `qapp` fixture to `tests/conftest.py`.**
- The preview is scaled with `Qt.KeepAspectRatio`.
- `status["message"]` keys are translated with `t()`.
- If `store.problems` is non-empty, show one message on start: "file set aside as .broken, defaults used".

- [ ] **Step 4: Run** the full suite. Expected: all pass. Then run `run.bat` once to see the window open with a live preview.

- [ ] **Step 5: Commit** `git add i18n.py ui.py tests && git commit -m "Add the Qt main window and Settings tab (ID/EN)"`

---

### Task 9: Jurus tab + editor

**Files:**
- Modify: `ui.py`, `i18n.py`, `tests/test_ui.py`

**Interfaces:**
- Consumes:
  - `rules.build`, `Store.import_media/save_jurus`, `BUILTIN_SOUNDS`, `BUILTIN_GESTURES`.
  - `Store.load_gestures` for the `rec:` choices.
- Produces:
  - `ui.JurusTab(store, engine)`:
    - A list showing ✓, name, gesture, sound and visual, with ⚠ + tooltip when `problem` is set.
    - Buttons Add/Edit/Delete/Try.
    - `apply()` saves, rebuilds with `rules.build` and calls `engine.reload`.
  - `ui.JurusEditor(QDialog)`, built as `JurusEditor(store, jurus: dict | None, taken: set[str])`, where `taken` ⊆ {"orb", "vanish"} is used by the other jurus.
    - Fields: name, gesture, sound (file / builtin, ▶ preview, mode, volume), visual (type, file or text + colour, position, size, held or seconds), orb sounds (4 pickers, visible for `orb`), hold, exclusive.
    - `value() -> dict` returns the jurus dict in the spec's schema.
    - Picking a file calls `store.import_media`.
    - File dialog filters:
      - sound `*.mp3 *.wav *.ogg *.flac`;
      - image `*.png *.jpg *.jpeg *.gif`;
      - video `*.mp4 *.mov *.webm *.avi`.

- [ ] **Step 1: Write the failing tests**

```python
def test_editor_round_trips_a_jurus(tmp_path, qapp):
    j = DEFAULT_JURUS[-1]                    # WOW text jurus
    assert JurusEditor(Store(tmp_path), j, set()).value() == j

def test_orb_and_vanish_are_greyed_out_when_taken(tmp_path, qapp):
    e = JurusEditor(Store(tmp_path), None, {"orb"})
    assert not e.visual_type.model().item(e.visual_type.findData("orb")).isEnabled()

def test_jurus_tab_flags_a_missing_file(tmp_path, qapp):
    s = Store(tmp_path); s.save_jurus([{**DEFAULT_JURUS[1], "sound": {"file": "media/gone.mp3", "mode": "once", "volume": 1.0}}])
    tab = JurusTab(s, make_engine(s))
    assert "⚠" in tab.list.item(0).text()
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_ui.py -q`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement.** Use `QFormLayout`. The ▶ preview plays through `engine.audio.play`. Add `# ponytail: media loads in the UI thread on save (a long clip can freeze the window ~1 s); move to a worker if it bites`.

- [ ] **Step 4: Run** the full suite. Expected: all pass.

- [ ] **Step 5: Commit** `git commit -am "Add the Jurus tab and editor"`

---

### Task 10: Gesture tab + recording

**Files:**
- Modify: `ui.py`, `i18n.py`, `tests/test_ui.py`

**Interfaces:**
- Consumes:
  - `Engine.record()` / `recorded` / `status["scores"]` (Task 7).
  - `Store.load_gestures/save_gestures`, `BUILTIN_GESTURES`.
- Produces:
  - `ui.GestureTab(store, engine)`. Contents:
    - The built-in list, read-only.
    - Recorded gestures with Rename/Delete/Re-record.
    - A Record button that asks for a name, then calls `engine.record()`.
    - A sensitivity slider (0.8–3.0) for the selected gesture.
    - A live score bar for the selected gesture, fed from `status["scores"]`.
  - `on_recorded(result)`:
    - A dict is named, saved, and triggers `JurusTab.apply()`.
    - An error key shows `t("record_" + key)` in a message box.
  - A rename updates `rec:<old>` → `rec:<new>` in `jurus.yaml`. A delete leaves the jurus in place; Task 6 flags it with ⚠.

- [ ] **Step 1: Write the failing tests**

```python
def test_recorded_gesture_is_saved_and_offered_in_the_editor(tmp_path, qapp):
    s = Store(tmp_path); tab = GestureTab(s, make_engine(s)); tab.pending_name = "peace"
    tab.on_recorded({"name": "", "hands": 1, "mean": [0.0] * 210, "radius": 0.05, "sensitivity": 1.5})
    assert s.load_gestures()[0]["name"] == "peace"
    assert JurusEditor(s, None, set()).gesture.findData("rec:peace") >= 0

def test_rename_updates_jurus_that_use_it(tmp_path, qapp): ...
def test_failed_recording_shows_a_message_and_saves_nothing(tmp_path, qapp, monkeypatch): ...  # patch QMessageBox.warning
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_ui.py -q`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run** the full suite. Expected: all pass.

- [ ] **Step 5: Commit** `git commit -am "Add the Gesture tab with recording and live match score"`

---

### Task 11: migrate this PC, docs, live check, PR

**Files:**
- Modify: `README.md` (run, tabs, jurus, recording, where data lives; drop the `config.yaml` editing sections for SFX/poses), `DECISIONS.md` (one entry per spec decision, dated 2026-10-10), `.gitignore` (`*.broken`)
- One-off, not committed: `<scratchpad>/migrate.py`

- [ ] **Step 1: Write and run `migrate.py`.** It builds `%APPDATA%\MeetCam\jurus.yaml` from today's `config.yaml` (from git `main`) plus `assets/`:

| Gesture | Name | Sound | Visual |
|---|---|---|---|
| OPEN_PALM | Orb | — | `orb`, `file` = the user's orb clip |
| ONE…FOUR | — | the four finger MP3s, mode `once` | — |
| SEAL | Seal | the user's seal sound (exclusive) | `vanish` with the user's smoke clip |
| WOW | Wow | `anime-wow-sound-effect.mp3` (exclusive) | `video` `wow.mp4`, position `full`, size 1.0, duration = clip length |
| RAISED_FIST | Raised fist | the user's raised-fist sound (exclusive) | `video` (the user's clip), position `bottom_left`, size 0.8, duration `held` |

Every media file goes in through `Store.import_media`. It also copies `assets/background.png` to `store.background`.

- [ ] **Step 2: Run** the full suite. Expected: all pass. Then run `run.bat` and confirm the window shows the migrated jurus with no ⚠.

- [ ] **Step 3: Update the docs and commit** `git commit -am "Docs for the app and jurus"`

- [ ] **Step 4: Live checklist for the user.** Report the results; fix failures before the PR.
  1. Preview and lights. Meet still receives video, voice and SFX.
  2. Each migrated jurus fires as before (orb charge/blast, fingers stop on release, seal vanish, wow, raised fist).
  3. Add a jurus of each kind (image PNG, GIF, video, text) at a grid position and at `hand`. Text reads correctly in both the preview and Meet.
  4. Record a one-hand gesture and a two-hand gesture, attach them to jurus, and check that they fire. Check the score bar and the sensitivity slider.
  5. Switch the language to EN and back. Restart: everything persists.

- [ ] **Step 5: Push and open the PR.** Run `git push -u origin app-customization`, then `gh pr create --base main`, with a summary and the checklist results.
