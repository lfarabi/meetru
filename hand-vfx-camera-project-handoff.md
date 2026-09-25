# Real-Time Hand VFX Camera — Project Handoff

## 1. Project Overview

Ide utama: membuat aplikasi desktop real-time yang menggunakan webcam untuk mendeteksi gerakan tangan, lalu memunculkan efek visual dan sound effect yang mengikuti gesture tangan.

Target penggunaan:
- Zoom
- Google Meet
- OBS
- Discord / livestream
- Platform lain yang menerima virtual camera dan virtual microphone

Contoh pengalaman:
- User membuka telapak tangan → energy orb muncul.
- Tangan bergerak → orb mengikuti telapak.
- Tangan mengepal → orb melakukan charging.
- Tangan didorong ke depan → energy blast.
- Setiap aksi mempunyai sound effect yang sinkron dengan visual.

Konsep akhirnya bukan sekadar webcam filter, tetapi:
> Gerakan tubuh → gesture recognition → command/state → visual + audio response.

---

# 2. Core Product Goal

Membangun **Real-Time Hand VFX Engine** yang:

1. Mengambil video dari webcam.
2. Melakukan hand tracking secara real-time.
3. Mendeteksi gesture.
4. Mengubah gesture menjadi state/command.
5. Merender VFX yang mengikuti tangan.
6. Memainkan sound effect berdasarkan gesture/state.
7. Menghasilkan video sebagai virtual camera.
8. Menghasilkan audio sebagai virtual microphone/audio device.
9. Dapat digunakan di Zoom dan Google Meet.

---

# 3. MVP

MVP pertama sengaja dibuat sederhana.

### Effect utama

**Energy Orb**

### Gesture MVP

| Gesture | Behavior | Audio |
|---|---|---|
| NO_HAND | Tidak ada effect | Tidak ada |
| OPEN_PALM | Spawn energy orb | Activation / whoosh |
| FIST | Orb charging | Energy hum |
| FIST + tahan | Orb semakin besar/kuat | Pitch/intensity meningkat |
| HAND PUSH | Energy blast | Blast / impact |
| NO_HAND setelah effect | Fade out | Power-down |

Target:
- 1280x720
- minimum 30 FPS
- ideal 60 FPS
- latency gerakan tangan → VFX rendah
- tracking stabil dan tidak jittery

---

# 4. Recommended Technology Stack

## Core

**Python 3.11+**

Alasan:
- cepat untuk prototyping
- ecosystem computer vision matang
- mudah dikembangkan dengan Claude Code
- integrasi OpenCV dan MediaPipe mudah

## Computer Vision / Hand Tracking

**MediaPipe Hand Landmarker / MediaPipe Tasks**

Target output:
- 21 landmarks per tangan
- koordinat x/y/z
- left/right hand jika tersedia

Landmark penting:
- wrist
- index MCP
- middle MCP
- ring MCP
- pinky MCP
- fingertips

## Rendering MVP

**OpenCV + NumPy**

Tidak menggunakan Unity/Unreal pada MVP.

Alasan:
- kebutuhan awal masih procedural VFX
- lebih cepat dibuat
- mudah debugging
- tidak menambah kompleksitas game engine

## Configuration

**YAML / PyYAML**

Semua parameter tuning sebaiknya berada di `config.yaml`.

## Video Output

Untuk MVP:
**OBS Studio + OBS Virtual Camera**

Architecture:

```text
Python VFX App
      ↓
OBS
      ↓
OBS Virtual Camera
      ↓
Zoom / Google Meet
```

Jangan membuat custom virtual-camera driver pada MVP.

## Audio Output

Gunakan virtual audio device / virtual microphone.

Contoh konsep:

```text
Physical Microphone
        ↓
        ├─────────────┐
        ↓             │
     Audio Mixer      │
        ↑             │
        │             │
      SFX Engine      │
        ↓             │
Virtual Audio Device ←┘
        ↓
Zoom / Google Meet
```

Teknologi virtual audio dapat disesuaikan dengan OS, misalnya:
- VB-CABLE / VoiceMeeter di Windows
- BlackHole di macOS

Untuk MVP, gunakan virtual audio device yang sudah tersedia daripada membuat custom audio driver.

---

# 5. High-Level Architecture

```text
                 ┌───────────────┐
                 │    Webcam     │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │ Hand Tracking │
                 │   MediaPipe   │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │    Gesture    │
                 │    Engine     │
                 └───┬───────┬───┘
                     ↓       ↓
              ┌──────────┐ ┌───────────┐
              │   VFX    │ │   Audio   │
              │  Engine  │ │   Engine  │
              └────┬─────┘ └─────┬─────┘
                   ↓             ↓
             Video Output    Audio Mixer
                   ↓             ↓
             OBS Virtual    Virtual Audio
               Camera          Device
                   ↓             ↓
                 ┌─────────────────┐
                 │  Zoom / Meet    │
                 └─────────────────┘
```

---

# 6. Hand Tracking Pipeline

Per frame:

```text
Camera Frame
      ↓
Resize / preprocess
      ↓
MediaPipe Hand Landmarker
      ↓
21 Landmarks
      ↓
Coordinate normalization
      ↓
Palm calculation
      ↓
Smoothing
      ↓
Gesture detection
      ↓
VFX state update
      ↓
Audio state update
      ↓
Render
```

---

# 7. Palm Position

Jangan menggunakan satu landmark saja.

Hitung palm center dari:

```text
wrist
index MCP
middle MCP
ring MCP
pinky MCP
```

Formula:

```python
palm = (
    wrist
    + index_mcp
    + middle_mcp
    + ring_mcp
    + pinky_mcp
) / 5
```

Kemudian konversi koordinat normalized MediaPipe ke pixel:

```python
x = landmark.x * frame_width
y = landmark.y * frame_height
```

---

# 8. Smoothing

Raw hand landmarks berpotensi jitter.

Gunakan exponential smoothing:

```python
smooth_x = alpha * current_x + (1 - alpha) * previous_x
smooth_y = alpha * current_y + (1 - alpha) * previous_y
```

Contoh initial value:

```yaml
smoothing: 0.35
```

Parameter harus configurable.

Tujuan:
- mengurangi jitter
- tetap responsif
- tidak membuat VFX terasa terlambat

---

# 9. Gesture Detection

MVP tidak membutuhkan ML classifier tambahan.

Gunakan geometric heuristics.

## OPEN_PALM

Jika sebagian besar fingertip berada jauh dari palm:

```text
index tip
middle tip
ring tip
pinky tip
```

→ `OPEN_PALM`

## FIST

Jika fingertip relatif dekat dengan palm:

→ `FIST`

## NO_HAND

Tidak ada hand yang terdeteksi:

→ `NO_HAND`

Future gestures:

```text
POINT
PEACE
THUMBS_UP
PINCH
HAND_PUSH
SNAP
```

Gesture tambahan tidak perlu dimasukkan ke MVP sebelum core tracking stabil.

---

# 10. VFX State Machine

Gunakan explicit state machine.

```text
HIDDEN
   ↓
SPAWNING
   ↓
ACTIVE
   ↓
CHARGING
   ↓
ACTIVE
   ↓
DISAPPEARING
   ↓
HIDDEN
```

Behavior:

### NO_HAND

```text
HIDDEN
```

### OPEN_PALM

```text
HIDDEN → SPAWNING → ACTIVE
```

### FIST

```text
ACTIVE → CHARGING
```

### OPEN_PALM setelah charging

```text
CHARGING → ACTIVE
```

### NO_HAND

```text
ACTIVE / CHARGING → DISAPPEARING → HIDDEN
```

---

# 11. Energy Orb Design

MVP orb dibuat procedural.

Layer:

```text
outer glow
     ↓
energy ring
     ↓
core
     ↓
particle field
```

Teknik:
- radial gradient
- Gaussian blur
- alpha compositing
- additive-style blending
- procedural particles
- pulse animation

Tujuan visual:
- tidak terlihat seperti PNG yang ditempel
- terasa mengikuti tangan
- memiliki depth dan energy

---

# 12. Orb Position

Orb sebaiknya tidak tepat berada di tengah palm.

Gunakan arah:

```text
wrist → middle MCP
```

Kemudian:

```python
orb_position =
    palm_center
    + direction_from_wrist_to_middle_mcp * OFFSET
```

Sehingga visual:

```text
       ORB
        ●
       /
      /
    PALM
```

bukan:

```text
    ●
   PALM
```

---

# 13. Depth / Orb Size

Gunakan ukuran tangan sebagai proxy jarak dari kamera.

Contoh:

```text
hand_width =
distance(index_mcp, pinky_mcp)
```

Jika hand width membesar:

```text
tangan lebih dekat kamera
        ↓
orb scale meningkat
```

Behavior:

```text
tangan menjauh → orb mengecil
tangan mendekat → orb membesar
```

Jangan terlalu bergantung pada nilai absolut MediaPipe `z` pada MVP.

---

# 14. Charging Behavior

Saat FIST:

```text
0 sec → 1.0x
1 sec → 1.2x
2 sec → 1.5x
3 sec → 2.0x
```

Selain scale:
- glow meningkat
- particle activity meningkat
- brightness meningkat
- audio pitch/intensity meningkat

Contoh:

```text
OPEN PALM
   ↓
Energy Orb

FIST
   ↓
Charge
   ↓
Orb grows
   ↓
Particles increase
   ↓
Audio pitch rises
```

---

# 15. Audio Engine

Audio harus menjadi bagian dari architecture sejak awal.

Jangan hanya:

```text
gesture → play mp3
```

Gunakan:

```text
gesture
   ↓
state
   ↓
audio parameters
   ↓
SFX output
```

Contoh:

```text
OPEN_PALM
    ↓
activation.wav

FIST
    ↓
charging loop

charge duration
    ↓
pitch/intensity meningkat

HAND_PUSH
    ↓
blast.wav

NO_HAND
    ↓
power_down.wav
```

### Audio yang dibutuhkan

Minimal:

```text
activation.wav
charge_loop.wav
blast.wav
power_down.wav
```

Future:

```text
lightning.wav
magic_cast.wav
telekinesis.wav
impact.wav
whoosh.wav
```

---

# 16. Audio State Example

```text
✋ OPEN PALM
↓
"WHOOSH"

✊ FIST
↓
"MMMMMMMM..."

✊ tahan 1 sec
↓
pitch normal

✊ tahan 2 sec
↓
pitch naik

✊ tahan 3 sec
↓
pitch semakin tinggi

👉 PUSH
↓
"BOOOOM!"
```

Audio dan visual harus terasa sebagai satu event.

---

# 17. Zoom Integration

Video:

```text
Python App
   ↓
OBS
   ↓
OBS Virtual Camera
   ↓
Zoom
```

Di Zoom user memilih:

```text
Camera → OBS Virtual Camera
```

Audio:

```text
Microphone
+
SFX
↓
Virtual Audio Device
↓
Zoom Microphone
```

---

# 18. Google Meet Integration

Google Meet juga dapat menggunakan virtual camera/audio device.

Video:

```text
Python App
   ↓
OBS Virtual Camera
   ↓
Google Meet Camera
```

Audio:

```text
Microphone
+
SFX
↓
Audio Mixer
↓
Virtual Audio Device
↓
Google Meet Microphone
```

Google Meet tidak perlu mengetahui bahwa efek berasal dari Python. Ia hanya menerima virtual camera dan virtual microphone sebagai input.

---

# 19. Important Audio Consideration

SFX harus masuk ke audio input yang dipilih oleh Meet/Zoom agar peserta lain mendengarnya.

Recommended architecture:

```text
             ┌── Physical Microphone
             │
             ↓
        ┌─────────────┐
        │ Audio Mixer │
        └──────┬──────┘
               ↑
               │
          SFX Engine
               │
               ↓
      Virtual Microphone
               ↓
          Zoom / Meet
```

Jangan mengandalkan speaker output karena itu berisiko:
- echo
- feedback
- audio delay
- SFX tidak masuk ke meeting

---

# 20. Project Structure

```text
hand-vfx-camera/
│
├── README.md
├── PRD.md
├── requirements.txt
├── config.yaml
│
├── src/
│   ├── main.py
│   │
│   ├── camera/
│   │   └── webcam.py
│   │
│   ├── tracking/
│   │   ├── hand_tracker.py
│   │   ├── landmarks.py
│   │   └── smoothing.py
│   │
│   ├── gestures/
│   │   ├── detector.py
│   │   └── states.py
│   │
│   ├── effects/
│   │   ├── base.py
│   │   ├── energy_orb.py
│   │   ├── particles.py
│   │   └── compositor.py
│   │
│   ├── audio/
│   │   ├── engine.py
│   │   ├── mixer.py
│   │   └── states.py
│   │
│   ├── output/
│   │   ├── preview.py
│   │   ├── virtual_camera.py
│   │   └── virtual_audio.py
│   │
│   └── utils/
│       ├── config.py
│       └── math.py
│
├── assets/
│   ├── effects/
│   └── audio/
│
└── tests/
    ├── test_gestures.py
    ├── test_smoothing.py
    ├── test_math.py
    └── test_states.py
```

---

# 21. Configuration

Example `config.yaml`:

```yaml
camera:
  device_id: 0
  width: 1280
  height: 720
  fps: 30

tracking:
  confidence: 0.6
  smoothing: 0.35

effect:
  enabled: true
  orb_scale: 1.0
  glow: 1.0
  particles: 50

gesture:
  fist_threshold: 0.45
  open_palm_threshold: 0.55

audio:
  enabled: true
  master_volume: 1.0
  charge_volume: 0.7
  sfx_volume: 1.0
```

All tunable parameters should be configurable rather than hard-coded.

---

# 22. Performance Requirements

| Metric | Target |
|---|---:|
| Resolution | 1280×720 |
| FPS | ≥30 |
| Ideal FPS | 60 |
| Tracking latency | <100 ms |
| VFX latency | <50 ms |
| CPU | <40% target |
| RAM | <500 MB target |

Do not optimize prematurely.

Development order:

```text
functional
    ↓
stable
    ↓
fast
    ↓
beautiful
```

---

# 23. Development Phases

## Phase 1 — Webcam

Goal:

```text
Webcam → Python → Preview
```

Acceptance:
- webcam detected
- 1280×720
- ≥30 FPS

## Phase 2 — Hand Tracking

Goal:

```text
Webcam
 ↓
MediaPipe
 ↓
21 landmarks
```

Display hand skeleton.

Acceptance:
- detect hand
- palm coordinate available
- stable tracking

## Phase 3 — Smoothing

Goal:

```text
Raw landmark
 ↓
Low-pass filter
 ↓
Stable position
```

Acceptance:
- reduced jitter
- responsive movement

## Phase 4 — Gesture

Implement:

```text
OPEN_PALM
FIST
NO_HAND
```

Acceptance:

```text
✋ → OPEN_PALM
✊ → FIST
nothing → NO_HAND
```

## Phase 5 — Energy Orb

Implement:
- spawn
- follow palm
- glow
- particles
- alpha compositing

## Phase 6 — Charging

Implement:
- scale growth
- glow growth
- particle increase
- state transitions

## Phase 7 — Audio

Implement:
- activation
- charge loop
- pitch/intensity modulation
- blast
- power-down

## Phase 8 — OBS

Implement output through OBS Virtual Camera.

## Phase 9 — Audio Device

Configure virtual audio device / mixer.

## Phase 10 — Zoom + Google Meet Testing

Test:
- video
- audio
- latency
- echo
- CPU usage
- stability

---

# 24. Future Effects

Architecture should allow multiple effects:

```text
Effect
├── EnergyOrb
├── FireBall
├── Lightning
├── Orb
├── MagicCircle
├── Telekinesis
└── GlowingEyes
```

Suggested interface:

```python
class Effect:
    def update(self, hand_state, dt):
        pass

    def render(self, frame):
        pass
```

Effects should be replaceable without modifying hand tracking.

---

# 25. Future "Orb" Effect

After Energy Orb is stable, build Orb.

Visual layers:

```text
particles
     ↓
spiral particle field
     ↓
radial blur
     ↓
turbulent/noise texture
     ↓
energy core
     ↓
pulsing glow
```

The effect should not merely be a blue circle.

It should have:
- rotational particles
- spiral trails
- radial blur
- turbulence
- pulsing brightness
- hand-relative positioning
- charging animation

---

# 26. Potential Advanced Gesture System

Future gestures:

```text
OPEN_PALM
FIST
POINT
PEACE
THUMBS_UP
PINCH
HAND_PUSH
SNAP
TWO_HANDS
```

Potential mapping:

```text
OPEN PALM
→ Energy

FIST
→ Charge

POINT
→ Lightning

PINCH
→ Telekinesis

TWO HANDS
→ Magic circle

HAND PUSH
→ Blast
```

---

# 27. Product Direction

Potential long-term product concept:

**Real-Life Anime Controller**

User controls virtual effects through body movement.

Examples:

```text
Hand movement
→ Fire

Finger point
→ Lightning

Fist
→ Energy charge

Push
→ Blast

Two hands
→ Magic circle
```

The same engine can later be used for:
- Zoom entertainment
- Google Meet
- streaming
- TikTok/Instagram content
- OBS
- VTubing
- live events
- interactive presentations

---

# 28. Claude Code Initial Prompt

Use this as the initial prompt in Claude Code:

> You are a senior computer vision and real-time graphics engineer.
>
> Build a desktop prototype called `hand-vfx-camera`.
>
> ## Product Goal
>
> Create a real-time webcam application that detects a user's hand using MediaPipe Hand Landmarker and renders a visual effect that follows the user's palm.
>
> The processed video should eventually be usable as a virtual camera input for Zoom and Google Meet through OBS Virtual Camera.
>
> The application should also support a sound-effect engine driven by gesture/state. The final architecture must separate hand tracking, gesture recognition, VFX rendering, audio rendering, and output adapters.
>
> ## MVP
>
> Implement one effect:
>
> `Energy Orb`
>
> Gesture behavior:
>
> - NO_HAND → no effect
> - OPEN_PALM → spawn energy orb above the palm
> - FIST → orb enters charging state
> - Moving hand → orb smoothly follows the hand
> - FIST held longer → orb scale/glow/particle activity increase
> - HAND_PUSH can initially be implemented as a future extension; do not block MVP on it
> - Hand disappears → orb fades out
>
> Audio behavior:
>
> - OPEN_PALM → activation SFX
> - FIST → charging loop
> - longer charge → increasing pitch/intensity
> - future HAND_PUSH → blast SFX
> - NO_HAND → power-down SFX
>
> ## Technology
>
> Use:
>
> - Python 3.11+
> - OpenCV
> - MediaPipe Tasks / Hand Landmarker
> - NumPy
> - PyYAML
>
> Do not introduce Unity, Unreal, Electron, or a custom virtual-camera/audio driver in the MVP.
>
> Use OpenCV/NumPy for initial VFX rendering.
>
> ## Architecture
>
> Use:
>
> `camera`
> `tracking`
> `gestures`
> `effects`
> `audio`
> `output`
> `utils`
>
> Keep modules separated by responsibility.
>
> ## Hand Tracking
>
> Track 21 MediaPipe hand landmarks.
>
> Calculate palm center using:
>
> - wrist
> - index MCP
> - middle MCP
> - ring MCP
> - pinky MCP
>
> Apply exponential smoothing to palm coordinates.
>
> Make smoothing configurable through config.yaml.
>
> ## Gesture Detection
>
> Implement geometric heuristics for:
>
> - NO_HAND
> - OPEN_PALM
> - FIST
>
> Do not use an additional ML classifier for the MVP.
>
> Expose gesture detection as a separate module.
>
> ## Energy Orb
>
> Implement the orb procedurally using NumPy/OpenCV.
>
> It should contain:
>
> 1. core
> 2. radial glow
> 3. energy ring
> 4. particles
> 5. subtle pulsing animation
>
> Use alpha/additive-style compositing where appropriate.
>
> Position the orb slightly above the palm using the direction from wrist toward middle MCP.
>
> ## Charging
>
> When the gesture changes from OPEN_PALM to FIST:
>
> - enter CHARGING state
> - increase orb scale over time
> - increase glow
> - increase particle activity
>
> When returning to OPEN_PALM:
>
> - return to ACTIVE state
>
> When NO_HAND:
>
> - fade out and return to HIDDEN
>
> Implement this using an explicit state machine.
>
> ## Audio
>
> Implement an audio abstraction that supports:
>
> - one-shot SFX
> - looping charging sound
> - volume control
> - basic pitch/intensity control if practical
>
> Keep the audio engine independent from the VFX engine.
>
> Use placeholder WAV assets if necessary.
>
> Do not implement a custom virtual audio driver. Output integration will be handled through an OS virtual audio device.
>
> ## Performance
>
> Target:
>
> - 1280x720
> - minimum 30 FPS
> - low latency
> - avoid unnecessary frame copies
> - avoid blocking operations inside render loop
>
> Display:
>
> - FPS
> - current gesture
> - current VFX state
> - hand detected/not detected
>
> ## Configuration
>
> Create config.yaml containing:
>
> - camera settings
> - tracking confidence
> - smoothing factor
> - effect scale
> - glow
> - particle count
> - gesture thresholds
> - audio volume
>
> Do not hard-code tunable values.
>
> ## Testing
>
> Create unit tests for:
>
> - palm center calculation
> - smoothing
> - gesture classification
> - state transitions
>
> ## Development Strategy
>
> Implement incrementally:
>
> 1. webcam preview
> 2. hand landmarks
> 3. palm tracking
> 4. smoothing
> 5. gesture detection
> 6. energy orb
> 7. charging state
> 8. audio engine
> 9. performance instrumentation
> 10. output adapter preparation
>
> Do not implement OBS integration until the core pipeline works.
>
> ## Engineering Rules
>
> - Keep render loop simple.
> - Avoid unnecessary abstractions.
> - Prefer measurable performance over premature architecture.
> - Add logging for camera initialization and tracking failures.
> - Fail gracefully if no camera or MediaPipe model is available.
> - Document setup and execution in README.md.
> - Include exact commands to create the virtual environment, install dependencies, obtain the MediaPipe model, and run the application.
>
> After implementation, run tests and provide a concise report of:
>
> - what works
> - what remains
> - test results
> - performance
> - environment-specific limitations

---

# 29. Recommended Implementation Strategy

Do NOT ask Claude Code to build the entire finished product in one shot.

Use iterative milestones:

### Sprint 1

```text
Camera
+
MediaPipe
+
Landmarks
```

### Sprint 2

```text
Palm tracking
+
Smoothing
+
Gesture detection
+
State machine
```

### Sprint 3

```text
Energy Orb
+
Particles
+
Glow
```

### Sprint 4

```text
Audio engine
+
Gesture SFX
+
Charging audio
```

### Sprint 5

```text
OBS Virtual Camera
+
Virtual Audio
```

### Sprint 6

```text
Zoom
+
Google Meet
+
Latency
+
Echo testing
```

### Sprint 7

```text
Orb
+
Fire
+
Lightning
+
Magic Circle
```

---

# 30. Key Architectural Decision

The most important design decision:

**OBS and virtual audio devices should be treated as output adapters, not as the core application.**

Core:

```text
Hand Tracking
      ↓
Gesture Engine
      ↓
State Engine
      ↓
VFX + Audio
```

Output:

```text
                    ┌→ OBS Virtual Camera
VFX Engine ──────────┤
                    └→ future native camera output

                    ┌→ Virtual Audio Device
Audio Engine ────────┤
                    └→ future native audio output
```

This keeps the core engine reusable and prevents the project from becoming tightly coupled to Zoom, Google Meet, or OBS.
