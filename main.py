"""Hand VFX Camera: webcam → hand tracking → gesture → Energy Orb, with a mirrored debug preview."""
import logging
import sys
import time

import cv2
import yaml

from audio import AudioEngine
from gestures import FIST, HIDDEN, OPEN_PALM, GestureClassifier, OrbMachine
from orb import EnergyOrb, orb_anchor
from tracking import CONNECTIONS, HandTracker, Smoother

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hand-vfx")

GESTURE_COLORS = {OPEN_PALM: (80, 220, 80), FIST: (60, 140, 255)}


def open_camera(cfg: dict) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg["device_id"], cv2.CAP_DSHOW)
    # MJPG must be set before the size, otherwise most webcams fall back to slow raw YUY2 at 720p.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
    return cap


def draw_skeleton(img, lm):
    pts = lm[:, :2].astype(int)
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 4, (0, 200, 255), -1, cv2.LINE_AA)


def main() -> int:
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cap = open_camera(cfg["camera"])
    ok, frame = cap.read()
    if not ok:
        log.error("Camera %s could not be opened; check camera.device_id in config.yaml "
                  "and that no other app (Meet, Zoom, NVIDIA Broadcast) is using it", cfg["camera"]["device_id"])
        return 1
    log.info("Camera opened: %dx%d @ %.0f fps requested", frame.shape[1], frame.shape[0], cap.get(cv2.CAP_PROP_FPS))

    try:
        tracker = HandTracker(cfg["tracking"])
    except Exception as e:
        log.error("Hand tracker failed to start: %s", e)
        return 1
    smoother = Smoother(cfg["tracking"]["smoothing"])
    classifier = GestureClassifier(cfg["gesture"])
    machine = OrbMachine(cfg)
    orb = EnergyOrb(cfg)
    anchor = None  # last known orb position/radius, kept so the orb can fade out where the hand was
    effects_on, show_skeleton = True, False
    audio = AudioEngine(cfg["audio"])
    audio_state = HIDDEN  # state last reported to audio; forced to HIDDEN while effects are off

    fps, last = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("Camera frame dropped")
            continue
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        raw = tracker.detect(frame, int(now * 1000))
        lm = smoother.update(raw)
        gesture = classifier.update(lm)
        state = machine.update(gesture, classifier.openness if lm is not None else None, now)
        if lm is not None:
            anchor = orb_anchor(lm, cfg["orb"])
        if effects_on and anchor is not None:
            orb.draw(frame, machine, *anchor, now)
        new_audio_state = state if effects_on else HIDDEN
        audio.update(audio_state, new_audio_state, machine.charge, machine.blast_power)
        audio_state = new_audio_state

        # `frame` is what will go to the meeting (unmirrored, no overlay); preview is a mirrored copy.
        preview = frame.copy()
        if show_skeleton and lm is not None:
            draw_skeleton(preview, lm)
        preview = cv2.flip(preview, 1)
        color = GESTURE_COLORS.get(gesture, (180, 180, 180))
        cv2.putText(preview, f"FPS {fps:4.1f}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, gesture, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        status = f"{state}  charge {machine.charge:.0%}" if effects_on else "EFFECTS OFF"
        if audio.mixer.muted:
            status += "  [SFX MUTED]"
        cv2.putText(preview, status, (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if lm is not None:
            cv2.putText(preview, f"openness {classifier.openness:.2f}", (16, 142),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Hand VFX Camera (Space effects, M mute, D skeleton, Q quit)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            effects_on = not effects_on
        if key == ord("d"):
            show_skeleton = not show_skeleton
        if key == ord("m"):
            audio.mixer.muted = not audio.mixer.muted

    audio.close()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
