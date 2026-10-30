# Meetru

**Meetru** (from *Meeting Seru*, "fun meetings" in Indonesian): webcam hand tracking → gesture → sound and visual effects, sent to Google Meet / Zoom through a virtual camera. Everything you see and hear is a **jurus** ("move"): one hand gesture plus a sound and/or a visual, made and changed in the app. Design decisions: [DECISIONS.md](DECISIONS.md). Made by lfarabi with AI help: see [How Meetru was made](#how-meetru-was-made), [Privacy](#privacy) and [Credits, sources and licenses](#credits-sources-and-licenses).

## Setup (once)
```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
`requirements.txt` includes PySide6 (the Qt window) and Pillow (images and GIFs). The virtual camera needs OBS Studio installed once (it brings the "OBS Virtual Camera" driver; OBS itself need not run). For sound in meetings see [Use in Meet / Zoom](#use-in-meet--zoom).

## Start
Double-click `run.bat`, or run `.venv\Scripts\python main.py`. A webcam can be used by one app at a time: close NVIDIA Broadcast / Meet / Zoom if they hold the camera.

## The window
The live preview is on the left (mirrored, like a mirror; the meeting gets the unmirrored picture). Under it:

- Lights: **Virtual camera** and **Meeting audio** (ON/OFF; hover an OFF light to see what to install), FPS, the gesture seen right now, and an *Invisible* badge while the vanish jurus has hidden you.
- Buttons, each with a key: **Effects** (`Space`, all jurus on/off), **Mute sounds** (`M`, silences the effect sounds for you and the meeting, never your voice), **Room photo** (`B`), **Skeleton** (`D`, draws the tracked hand). `Q` quits. **About** (`F1`) shows the copyright, the license and the libraries used.

On the right are three tabs. Every change in them is saved at once and applies live; nothing needs a restart except the language.

### Jurus tab
The list shows each jurus as `name — gesture · sound · visual`. Tick the box to enable or disable one. A **⚠** in front means the jurus can't run (hover it for the reason): its recorded gesture was deleted, or its sound / picture / video file can't be read. It stays in the list and is skipped until you fix or delete it; everything else keeps working.

Buttons: **Add**, **Edit** (or double-click a row), **Delete** (asks first), **Try** (fires the selected jurus now, no gesture needed; not for the orb, which only its gesture summons).

The editor has:

- **Name** (must be unique) and **Hand gesture**: a built-in one or one you recorded.
- **Sound**: *(none)*, one of the built-in sounds (`activation`, `charge_loop`, `blast`, `power_down`, `finger_1` to `finger_4`) or **File...** (mp3, wav, ogg, flac; ▶ plays a preview on your speakers only, not in the meeting). **Play** is *Once* (stops early if you let go of the gesture, unless the jurus is exclusive) or *Repeat while the gesture is held*. Volume.
- **Visual**: *(none)*, **Picture** (png, jpg, gif), **Video**, **Text** (any text or emoji, a colour), **Energy orb** or **Vanish** (see below). Pictures, videos and text have a **Position**, a **Size** and **Shown for**: *While the gesture is held*, or a number of seconds.
- **Hold the gesture for**: how long it must be held before the jurus fires (0.3 s by default).
- **Exclusive**: the sound plays to the end and nothing else fires until the sound and the visual are done. A looping sound can't be exclusive.

A jurus fires once per hold of its gesture and re-arms when you let go. One sound plays at a time: a new sound cuts the previous one unless that one is exclusive. Two hands in view (and for 0.6 s after) count as a two-hand pose, so one-hand jurus wait. While the orb charges or blasts, or while the vanish runs, other jurus wait too.

Files you pick are copied into the data folder (see below), so moving or deleting the original does no harm.

### Gestures tab
Lists the built-in gestures and the ones you recorded.

| Built-in gesture | How to make it |
|---|---|
| 1, 2, 3, 4 fingers | that many fingers up, thumb folded |
| Open palm | open hand, fingers up and thumb out |
| Fist | closed hand |
| Raised fist | fist with the wrist in the upper 60% of the frame |
| Seal (2 hands) | two fingers up on each hand, one hand upright and one flat, crossed |
| Wow pose (2 hands) | both open palms together under the chin, fingers up (don't overlap the hands too much: the tracker may then see only one) |

To record your own: **Record**, type a name, then get ready: a 3 s countdown shows in the message line under the preview, then hold the pose still for 2 s. One hand or two hands is picked up from what the camera sees. The new gesture shows up in the jurus editor with a hand icon. **Re-record**, **Rename** (jurus using it follow the new name) and **Delete** (jurus using it stay, marked ⚠).

Select a recorded gesture and the **score bar** shows live how close your hand is to it; it fills and says *match* when the gesture is recognised. The **sensitivity** slider (0.8x to 3.0x, 1.5x by default) sets how strict: higher means easier to match. A recording knows the hand's shape, not which way it points (thumbs-up and thumbs-down look the same).

### Settings tab
Camera, microphone, speaker (what you hear), meeting output (what the others hear), send to the virtual camera on/off, language (Auto, Bahasa Indonesia, English; takes effect after a restart), mirror the preview. Devices switch at once.

## Visual types and positions
| Visual | What it does |
|---|---|
| Picture / Video / Text | drawn at its position and size, for as long as *Shown for* says |
| Energy orb | an anime-style energy orb: the gesture summons it on your hand. Hold it with the palm turned away from the camera to charge (up to 3 s, the sound rises), then turn the palm to the camera to blast. Changing to any other gesture, or taking the hand out of frame, puts it out. A video file is optional (a clip on a black backdrop, orb in the centre; without one a drawn orb is used), and so are the four orb sounds (appear, charge, blast, power down). |
| Vanish | a disappearing trick: the gesture swaps you for the empty-room photo, with a smoke clip (optional) bursting where your hands are; do it again and you come back |

Positions: *on the hand* (centred on the palm; size is in palm lengths), *full screen* (the clip keeps its own framing, so Size has no effect there), or one of nine spots on a 3x3 grid. Size is a fraction of the frame height. Positions are **as seen in your mirrored preview**: "left" is the left of your preview, and the meeting sees it on the right so that text and pictures read correctly there too (with the preview not mirrored, it shows what the meeting sees, positions included). There can be only one orb and one vanish jurus (the editor greys the option out when it is taken).

## Effect videos
A video used as a visual, as vanish smoke, or as the orb should have a **one-colour backdrop**: green or blue screen, black, white or another solid colour. The backdrop is read from the video's edges and removed automatically (black suits glowing effects; very dark parts of the subject can turn slightly see-through). A video **without** a one-colour backdrop is shown **as it is**, unedited: there is no AI cut-out.

## Vanish and the room photo
The vanish needs a photo of the empty room: press **Room photo** (`B`), step out of the frame within 5 s, and wait for *Empty-room photo saved*. Take it again whenever the camera or the light changes. While the vanish plays, its sound can't be cut and other jurus wait.

## Where your data lives
Everything you make is in your user data folder, `%APPDATA%\Meetru` (type it in the Explorer address bar):

| Path | Content |
|---|---|
| `jurus.yaml` | your jurus |
| `gestures.json` | your recorded gestures |
| `settings.yaml` | devices, language, mirror, skeleton |
| `media/` | the sounds, pictures and videos you picked (a clash in names gets `_2`, `_3`...; deleting a jurus leaves its files) |
| `background.png` | the empty-room photo |

Files are saved safely (a crash can't leave half a file). A file that can't be read is renamed to `<name>.broken`, the defaults are used, and the window tells you at start. On the first run you get a starter set with no copyrighted media: Bola Energi (a drawn energy orb), Beep 1-4 on the finger gestures, a plain vanish and a WOW text.

`config.yaml` stays as the advanced, technical tuning (gesture thresholds, orb look, audio latency). Normal use never needs it; the device, language and jurus choices above are not read from it.

**Media is never committed.** The media folders (`assets/audio`, `assets/video`) and `assets/background.png` are git-ignored, and your own media lives in `%APPDATA%` outside the repo.

## Use in Meet / Zoom
Once: install **VB-CABLE** (https://vb-audio.com/Cable/ → download the zip → right-click `VBCABLE_Setup_x64.exe` → Run as administrator → Install Driver → restart). Keep the Windows default output on your speakers or headphones, not "CABLE Input".

1. Close NVIDIA Broadcast and anything else holding the webcam.
2. Start Meetru **first**. Both lights should read ON: *Virtual camera* and *Meeting audio*.
3. In **Settings** pick your real **Microphone** (it is mixed into the meeting) and leave **Meeting output** on *CABLE Input*.
4. In Meet: ⋮ → Settings →
   - **Video** → Camera: **OBS Virtual Camera** (not your physical webcam: this app is using it)
   - **Audio** → Microphone: **CABLE Output (VB-Audio Virtual Cable)**; if effect sounds are clipped or missing, turn *Noise cancellation* off
5. Wear **headphones**: effect sounds also play on your speakers, and speakers feed back into the mic.

Your voice always passes through; **Mute sounds** only silences the effects. To mute your voice use Meet's own button. If Windows' default microphone is "CABLE Output" the app refuses it (it would hear itself), so pick the real one in Settings. Meet mirrors your own self-view; the others see the normal version, so text reads correctly. The skeleton and other preview-only drawings are never sent. If OBS Studio is open with *Start Virtual Camera* on, stop it first (the light will say OFF).

## Tests
```bash
.venv\Scripts\python -m pytest -q
```

## Privacy

- Your camera picture, your voice and your hand movements are processed on your own computer. Meetru does not upload them and has no account, server or analytics of its own.
- The hand-tracking library, Google's **MediaPipe**, sends performance and usage metrics about its own functions to Google (not your camera images; Google says input data stays on the device). See Google's [MediaPipe Tasks privacy notice](https://developers.google.com/edge/mediapipe/solutions/tasks), also printed at the top of MediaPipe's `NOTICE` file. By using Meetru you accept this; if you pass Meetru on to others, tell them.
- The first time you start Meetru it shows this and asks you to continue; if you choose Quit, nothing starts. **About** repeats it.
- What you show and play in a meeting is seen and heard by everyone in it.

## How Meetru was made

Meetru is a **vibe-coded** project by **lfarabi**. The idea, the requirements, the testing and every decision are mine. Most of the code and documentation were written by an AI coding assistant, **Claude Code** (Anthropic), working under my direction, with help from many open-source projects, tools and their documentation, all listed below. Commits written with the AI carry a `Co-Authored-By: Claude …` line, and the design notes in [DECISIONS.md](DECISIONS.md) and [docs/superpowers/](docs/superpowers/) record how the work was directed.

Meetru is **not a fork** of another project. It was written from scratch for this repository and, to the best of my knowledge, contains no code copied from other projects. The libraries it uses are installed separately (from PyPI, or downloaded by you) and are not part of this repository; their own licenses apply to them.

## Credits, sources and licenses

Libraries Meetru uses (installed with `requirements.txt`):

| Library | Used for | License |
|---|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Google) | hand tracking | Apache-2.0 |
| [MediaPipe Hand Landmarker model](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) (`hand_landmarker.task`, Google) | the trained hand model downloaded in Setup | Apache-2.0 (per Google's model card) |
| [OpenCV](https://opencv.org) ([opencv-contrib-python](https://github.com/opencv/opencv-python)) | camera capture and image processing; the package includes FFmpeg (LGPL-2.1-or-later) and Intel IPP (Intel Simplified Software License) | Apache-2.0 |
| [Qt 6](https://www.qt.io) through [PySide6](https://pypi.org/project/PySide6/) (The Qt Company) | the window; the package includes FFmpeg (LGPL-2.1-or-later) | LGPL-3.0 (or GPL-3.0) |
| [NumPy](https://numpy.org) | numbers and images | BSD-3-Clause |
| [Pillow](https://pillow.readthedocs.io) | pictures and GIFs; based in part on the work of the FreeType Team and the Independent JPEG Group | MIT-CMU |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) with [PortAudio](https://www.portaudio.com) | playing sounds and reading the microphone | MIT |
| [soundfile](https://github.com/bastibe/python-soundfile) with [libsndfile](https://libsndfile.github.io/libsndfile/) | reading MP3, WAV, OGG and FLAC | BSD-3-Clause; libsndfile LGPL-2.1-or-later |
| [PyYAML](https://pyyaml.org) | the settings files | MIT |
| [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) (Maik Riechert and contributors) | sending the video to the OBS virtual camera | GPL-2.0 (see the note below) |
| [pytest](https://pytest.org) | the tests only | MIT |

They bring their own dependencies (matplotlib, absl-py, flatbuffers, certifi and others). The full list with versions, the components bundled inside each package, and what each license asks for is in the [license and IP audit](docs/research/2026-10-30-license-ip-audit.md).

**Note on pyvirtualcam.** It is licensed GPL-2.0 without "or later", which on the Free Software Foundation's reading cannot be combined with GPL-3.0 and Apache-2.0 code in one distributed package. This repository only contains Meetru's own source code, so nothing is combined here; before any installer that bundles the libraries is published, this has to be settled (options are in section 3 of the audit).

Installed by you, not included in Meetru:

- [OBS Studio](https://obsproject.com) by the OBS Project (GPL-2.0-or-later): provides the "OBS Virtual Camera" device.
- [VB-CABLE](https://vb-audio.com/Cable/) by VB-Audio Software: donationware (www.vb-cable.com); use inside a company or organisation needs a paid license from VB-Audio.
- [Python](https://www.python.org) (PSF License).

Tools used to build Meetru (not included in it):

- [Claude Code](https://claude.com/claude-code) by Anthropic: the AI coding assistant.
- [superpowers](https://github.com/obra/superpowers) by Jesse Vincent (MIT): planning, review and test-first workflow skills.
- [ponytail](https://github.com/dietrichgebert/ponytail) by Dietrich Gebert (MIT): "keep it simple" coding rules.
- [Matt Pocock's skills](https://github.com/mattpocock/skills): grilling, research and other workflow skills.
- [GitHub CLI](https://cli.github.com).

**Trademarks and inspiration.** Google Meet is a trademark of Google LLC; Zoom is a trademark of Zoom Communications, Inc.; Windows is a trademark of Microsoft Corporation; OBS Studio is a project of the OBS Project; VB-CABLE and VB-Audio are names of VB-Audio Software. They are mentioned only to say what Meetru works with. Meetru is an independent project and is not affiliated with or endorsed by any of them. The energy-orb and hand-seal effects are inspired by anime in general; Meetru is not affiliated with any anime or manga or their rights holders.

**Media.** Meetru ships no copyrighted media: the starter sounds are synthesized in code and the orb is drawn in code. The pictures, sounds and videos you add stay on your computer (`%APPDATA%\Meetru\media`) and are never part of this repository. Only use media you have the right to use, and don't use a real person's voice, face or name without their permission.

## License

Copyright © 2026 lfarabi.

Meetru is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version (SPDX: `GPL-3.0-or-later`; see [LICENSE](LICENSE)). It comes with **no warranty**. The libraries and tools listed above keep their own licenses.

Contributions are welcome under the same license: see [CONTRIBUTING.md](CONTRIBUTING.md) (commits are signed off with the [Developer Certificate of Origin](DCO)).
