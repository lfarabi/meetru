# MeetCam app + customization: design

Date: 2026-10-10. Sub-project 1 of 2. Sub-project 2 (macOS/Linux support, driver setup wizard,
installers built on GitHub Actions, making the repo public) gets its own spec after this ships.

## Goal

Turn the hand-VFX camera into a desktop app a non-technical person can use: one Qt window with the
live preview and settings, where every gesture → sound → visual pairing ("jurus") can be added,
changed or removed without touching a file, including gestures the user records themselves.
Everything that works today keeps working. Still started with `run.bat` on this PC until
sub-project 2.

Decisions taken with the user (2026-10-10):
- audience: free public release (repo goes public in sub-project 2, GPL-3.0)
- gestures: built-in list **and** user-recorded poses
- visuals: image (PNG/JPG/GIF), video clip, text, built-in specials (Orb, vanish)
- UI language: Indonesian + English, default from the OS locale
- UI toolkit: Qt (PySide6), one window

## Concepts

**Jurus** (EN: "move"): one gesture plus what it does.

```yaml
# jurus.yaml: a list, evaluated in order
- name: Orb
  enabled: true
  gesture: OPEN_PALM          # built-in id, or rec:<name> for a recorded gesture
  hold: 0.3                   # seconds the gesture must be held before the jurus fires
  exclusive: false            # true: its sound plays to the end and can't be cut; other jurus wait
  sound:                      # optional
    file: media/boom.mp3      # media/<file> or builtin:<name> (synthesized placeholders)
    mode: once                # once | loop (repeats while the gesture is held)
    volume: 1.0
  visual:                     # optional
    type: image               # image | video | text | orb | vanish
    file: media/sticker.gif   # image/video; orb: optional clip; vanish: optional smoke clip
    text: "WOW!"              # text only
    color: "#ffffff"          # text only
    position: top             # hand | full | top_left | top | top_right | left | center | right
                              #   | bottom_left | bottom | bottom_right  (as seen in the mirrored preview)
    size: 0.4                 # fraction of frame height (hand: in palm lengths)
    duration: held            # held | seconds as a number, e.g. 2.0
  orb_sounds:                 # orb only, each optional: activation, charge_loop, blast, power_down
    blast: media/orb-blast.mp3
```

**Built-in gesture ids** (existing classifiers, tuning unchanged): `ONE`, `TWO`, `THREE`, `FOUR`,
`OPEN_PALM`, `FIST`, `RAISED_FIST`, `SEAL` (two hands), `WOW` (two hands).

**Recorded gesture**: `rec:<name>`, a template made from ~2 s of the user holding a pose, one or
two hands.

## Behaviour rules (carried over from today)

- One sound at a time: a jurus's sound cuts the previous one, unless that one is `exclusive`.
- Non-exclusive sounds stop when the gesture is released; exclusive ones play to the end and block
  other jurus from firing until both their sound and visual have finished (today: seal, wow,
  Figure fist).
- Two hands seen within `gesture.two_hand_grace` → one-hand gestures don't fire.
- Orb charging/blast/cooldown → other jurus don't fire.
- A jurus fires once per hold; re-arms after release; short tracker gaps (`gap`) don't count as release.
- At most one `orb` jurus and one `vanish` jurus (the editor greys the option out when taken).
- Orb: the jurus's gesture summons it (was hard-wired OPEN_PALM); charge = palm turned away,
  blast = palm aimed at camera, as now. The FOUR-while-charging tolerance applies only when the
  trigger is OPEN_PALM.
- Vanish: fires toggle invisible/visible; smoke lands on the centre of the hands that made the
  gesture; needs the empty-room photo (button in the UI, same 5 s delay + median as B today).

## Recorded gestures (`poses.py`)

- Descriptor per hand: the 210 pairwise distances between the 21 world landmarks, divided by palm
  size (wrist → middle knuckle). Invariant to position, distance, rotation and left/right hand.
- Two hands: both hands' descriptors (compared in both orders, best kept) plus wrist-to-wrist
  distance in palm sizes (from pixel landmarks, since world landmarks are per-hand).
- Recording: countdown 3 s, then 2 s of frames (~60). Template = mean descriptor; radius =
  95th percentile of the samples' RMS distance to the mean, floored so a perfectly still recording
  isn't impossible to match.
- Matching: RMS distance to each template; match when distance < radius × sensitivity (slider,
  default 1.5). Score shown live = 1 − distance / threshold. Nearest matching template wins.
- A recorded match outranks the built-in finger/fist classification for that hand (more specific).
- Known limit: shape only, not direction (thumbs-up = thumbs-down). `ponytail:` note in code;
  add palm-normal to the descriptor if users ask.
- Stored in `gestures.json`: name, hands (1|2), mean, radius, sensitivity.

## Visuals (`overlay.py`)

| type | loading | drawing |
|---|---|---|
| image | Pillow: PNG/JPG; GIF frames + per-frame durations; alpha kept | alpha blend |
| video | existing `jutsu.load_clip` (one-colour backdrop keyed, else as-is) | alpha blend, loops while held |
| text | QPainter renders to an ARGB image (any font, emoji, outline for readability), cached | alpha blend |
| orb | existing `EnergyOrb` / `OrbMachine` | unchanged |
| vanish | existing `Vanish` | unchanged |

Placement: `hand` = centred on the palm, scaled by palm size; `full` = fit the frame; grid
positions with a margin; `size` scales. `duration: held` shows while the gesture is held (fade
out on release); a number shows that many seconds.

Mirroring: the meeting gets the unmirrored frame, the preview is mirrored. So image/text/video
overlays are drawn on both: the scene (camera + orb + vanish) is copied for the preview first,
overlays go on the outgoing frame at the mirrored x, and on the flipped preview at the preview x.
Text and images read correctly in both, and "left" means the left of the preview.

## Storage (`store.py`)

Data folder from `QStandardPaths.AppDataLocation`, app name `MeetCam`
(Windows `%APPDATA%\MeetCam`, macOS `~/Library/Application Support/MeetCam`,
Linux `~/.local/share/MeetCam`):

- `settings.yaml`: camera, mic, monitor speaker, meeting output, virtual camera on/off,
  language, mirror preview, skeleton. Overrides the matching `config.yaml` keys.
- `jurus.yaml`, `gestures.json`
- `media/`: files the user picks are copied in (name kept; a different file with the same name
  gets a numeric suffix). Deleting a jurus leaves its media (may be shared). `ponytail:` note,
  add cleanup if the folder grows.
- `background.png`: the empty-room photo.

Saved on every edit (write to temp file + rename, so a crash can't leave half a file). A file that
fails to parse is renamed `*.broken` and defaults are used, with a message in the UI.

`config.yaml` stays as the technical defaults (thresholds, orb tuning). Its `finger_sfx`, `vanish`,
`wow` and `figure` sections are replaced by jurus.

**First-run defaults** (no copyrighted media): Orb (OPEN_PALM → procedural orb, synthesized
orb sounds); Beep 1–4 (ONE…FOUR → synthesized finger sounds, FOUR looped); Menghilang
(SEAL → vanish, no smoke clip = plain crossfade); WOW (WOW → text "WOW!" top, 2 s).

**This PC**: the current setup (Anime smoke + sound, wow clip, Figure fist, finger MP3s,
orb.mp4) is converted once into this user's `jurus.yaml` + `media/` by a one-off script that
isn't committed.

## UI (`ui.py`, `i18n.py`)

Main window: preview on the left (mirrored, scaled to fit) with status lights (virtual camera,
meeting audio, FPS) and buttons (Effects on/off, Mute SFX, Room photo, Skeleton; shortcuts
Space/M/B/D/Q kept). Right side, three tabs:

- **Jurus**: list with enable checkbox, name, gesture, sound, visual; ⚠ when a file is missing.
  Add / Edit / Delete / Try (fires it now, no gesture needed).
- **Gesture**: built-in list (read-only) and recorded ones (rename, delete, re-record); Record
  button with countdown overlay on the preview; live match score for the selected gesture.
- **Settings**: camera (names from Qt, mapped to the OpenCV index), mic, speaker, meeting output,
  virtual camera on/off, language, mirror preview.

Jurus editor (dialog): Gesture → Sound (file picker, ▶ preview, mode, volume) → Visual (type,
file/text/colour, position, size, duration; orb shows its four sound pickers) → hold, exclusive.
Changes apply live, no restart.

Strings in `i18n.py` as two dicts (id, en); `t(key)`; default from `QLocale.system()`.

## Architecture

- `main.py`: starts the Qt app.
- `engine.py`: today's per-frame loop moved into a `QThread`. Emits the preview as a `QImage`
  (queued signal) plus a status dict. UI → engine: attribute flags (effects on, mute, skeleton) and
  `reload(jurus, gestures, settings)`, which builds the new rule objects in the UI thread and swaps
  one reference that the engine picks up on its next frame. Device changes reopen just that device.
- `rules.py`: per frame, gesture ids active → `HoldTrigger` per jurus → fire/release → audio and
  visual actions; enforces the behaviour rules above.
- `audio.py`: `AudioEngine` gets `set_sounds(dict)` so jurus sounds swap at runtime, and loop
  sounds become per-sound instead of a fixed tuple. Mic → VB-CABLE path unchanged.
- `jutsu.py`: keeps `load_clip`, `Vanish`, `HoldTrigger`, `seal_point`, `wow_pose`,
  `raised_fist`; `PoseClip` is replaced by `overlay.py`.
- `tracking.py`, `gestures.py`, `orb.py`: only the trigger generalisation above.
- New dependency: PySide6 (LGPL). Pillow is already installed (via mediapipe → matplotlib); it is
  added to requirements.txt because we now import it.

## Errors

- Camera fails to open → message in the preview area, pick another camera in Settings.
- No virtual camera / VB-CABLE → status light OFF with a tooltip saying what to install.
- Missing or unreadable media → that jurus shows ⚠ and is skipped; everything else runs.
- Nothing in a jurus or gesture file can crash the app.

## Testing

- Existing 77 tests kept, adjusted to the refactor.
- New, in `tests/`: recorded-gesture match survives rotation/scale/mirroring of a synthetic hand
  and rejects a different pose; rules fire once per hold, cut the previous sound, exclusive
  blocks, two-hand grace blocks one-hand jurus; store round-trip, media copy + name clash, broken
  file fallback; overlay placement and mirror maths; text render produces a non-empty sprite.
- One offscreen smoke test (`QT_QPA_PLATFORM=offscreen`) builds the main window and the jurus
  editor.
- Live checks by the user with a checklist (each tab, record + use a gesture, each visual type,
  Meet still receives video + audio).

## Out of scope (sub-project 2 or later)

macOS/Linux camera, audio and virtual-camera backends; driver setup wizard; installers and CI;
public repo. Also not planned: several visuals per jurus, jurus chains, export/import of jurus
packs, direction-sensitive recorded gestures.
