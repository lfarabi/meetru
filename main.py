"""Hand VFX Camera: webcam → hand tracking → gesture → Energy Orb / vanishing jutsu, with a mirrored debug preview."""
import logging
import sys
import time

import cv2
import numpy as np
import pyvirtualcam
import yaml

from audio import AudioEngine
from gestures import FIST, FOUR, HIDDEN, NO_HAND, ONE, OPEN_PALM, THREE, TWO, GestureClassifier, OrbMachine
from jutsu import Vanish, raised_fist, seal_point, wow_pose
from orb import EnergyOrb, occlude_by_hand, orb_anchor
from tracking import CONNECTIONS, WRIST, HandTracker, Smoother, palm_normal, pick_primary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hand-vfx")

GESTURE_COLORS = {OPEN_PALM: (80, 220, 80), FIST: (60, 140, 255), ONE: (255, 170, 50),
                  TWO: (120, 230, 60), THREE: (40, 150, 255), FOUR: (230, 90, 180)}


def open_camera(cfg: dict) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg["device_id"], cv2.CAP_DSHOW)
    # MJPG must be set before the size, otherwise most webcams fall back to slow raw YUY2 at 720p.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
    return cap


def open_virtual_camera(cfg: dict, width: int, height: int, fps: int) -> pyvirtualcam.Camera | None:
    """The output Meet/Zoom pick as a camera. Missing or busy driver → preview-only, not a crash."""
    if not cfg["enabled"]:
        return None
    try:
        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, backend=cfg["backend"],
                                  fmt=pyvirtualcam.PixelFormat.BGR)
    except Exception as e:
        log.warning("Virtual camera unavailable, running preview-only: %s "
                    "(is OBS Studio's own Start Virtual Camera active? Stop it or close OBS)", e)
        return None
    log.info("Virtual camera: sending %dx%d to \"%s\"; select it as the camera in Meet/Zoom", width, height, cam.device)
    return cam


def facing(n) -> str:
    """Dominant palm direction for the debug overlay (left/right as seen in the mirrored preview)."""
    axis = int(np.argmax(np.abs(n)))
    return [("RIGHT", "LEFT"), ("UP", "DOWN"), ("CAMERA", "AWAY")][axis][int(n[axis] > 0)]


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
    world_smoother = Smoother(cfg["orb"]["world_smoothing"])
    normal, palm_side = None, 1.0
    classifier = GestureClassifier(cfg["gesture"])
    machine = OrbMachine(cfg)
    orb = EnergyOrb(cfg)
    anchor = None  # last known orb position/radius, kept so the orb can fade out where the hand was
    effects_on, show_skeleton = True, False
    pose_sounds = {"jutsu": cfg["vanish"], "wow": cfg["wow"], "figure": cfg["figure"]}
    audio = AudioEngine(cfg["audio"], cfg["finger_sfx"], cfg["meeting_audio"], pose_sounds)
    audio_state = HIDDEN  # state last reported to audio; forced to HIDDEN while effects are off
    vcam = open_virtual_camera(cfg["virtual_camera"], frame.shape[1], frame.shape[0], cfg["camera"]["fps"])
    vanish = Vanish(cfg["vanish"], frame.shape)
    wow = PoseClip(cfg["wow"], "Wow")
    figure = PoseClip(cfg["figure"], "Figure")
    straight = cfg["gesture"]["finger_straight"]
    two_hands_at = float("-inf")  # last time two hands were seen

    fps, last = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("Camera frame dropped")
            continue
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        hands = tracker.detect(frame, int(now * 1000))
        hand = pick_primary(hands, None if smoother.value is None else smoother.value[WRIST, :2])
        lm = smoother.update(hand.lm if hand else None)
        world = world_smoother.update(hand.world if hand else None)
        gesture = classifier.update(lm, world)
        occlusion = 0.0
        if lm is not None:
            normal, palm_side = palm_normal(world, palm_side)
        # Two hands up = a two-hand pose (seal, wow) forming or held. While crossed or touching,
        # MediaPipe often loses one hand for a few frames, so "two hands" lingers for a grace
        # period: the lone hand seen then is half of a pose, not a finger count, fist or orb.
        if len(hands) > 1:
            two_hands_at = now
        two_hands = now - two_hands_at < cfg["gesture"]["two_hand_grace"]
        orb_gesture = NO_HAND if vanish.busy or two_hands else gesture
        state = machine.update(orb_gesture, -float(normal[2]) if lm is not None else None, now)  # palm-to-camera
        if lm is not None:
            anchor = orb_anchor(lm, normal, cfg["orb"])
            o = cfg["orb"]
            occlusion = float(np.clip((normal[2] - o["occlusion_start"]) / (o["occlusion_full"] - o["occlusion_start"]), 0, 1))
        vanish.capture_tick(frame, now)  # raw camera frame, before any effect
        if effects_on:
            if vanish.update(seal_point(hands, straight), now):
                audio.play_pose("jutsu")
            if wow.update(wow_pose(hands, straight), now):
                audio.play_pose("wow")
            fist_up = not two_hands and raised_fist(gesture, lm, frame.shape[0], cfg["figure"]["raise_above"])
            if figure.update(fist_up, now):
                audio.play_pose("figure")
            vanish.apply_plate(frame)
        if effects_on and anchor is not None:
            # an invisible hand doesn't cover the orb
            original = frame.copy() if occlusion > 0.01 and state != HIDDEN and vanish.mix < 0.5 else None
            orb.draw(frame, machine, *anchor, now)
            if original is not None:
                occlude_by_hand(frame, original, lm, occlusion)
        if effects_on:
            vanish.draw_smoke(frame, now)
            wow.draw(frame, now)
            figure.draw(frame, now)
        new_audio_state = state if effects_on else HIDDEN
        audio.update(audio_state, new_audio_state, machine.charge, machine.blast_power, machine.silent_fade)
        audio_state = new_audio_state
        # two hands up = a seal in progress, not a finger-count gesture
        audio.update_fingers(gesture if effects_on else NO_HAND, now,
                             machine.busy(now) or two_hands or vanish.busy or wow.busy or figure.busy)

        # `frame` is what goes to the meeting (unmirrored, no overlay); preview is a mirrored copy.
        if vcam is not None and frame.shape[:2] == (vcam.height, vcam.width):
            vcam.send(frame)
        preview = frame.copy()
        if show_skeleton and lm is not None:
            draw_skeleton(preview, lm)
        preview = cv2.flip(preview, 1)
        color = GESTURE_COLORS.get(gesture, (180, 180, 180))
        vcam_label = "VCAM ON" if vcam is not None else "VCAM OFF"
        meet_label = "MEET AUDIO ON" if audio.meeting_on else "MEET AUDIO OFF"
        cv2.putText(preview, f"FPS {fps:4.1f}  {vcam_label}  {meet_label}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, gesture, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        status = f"{state}  charge {machine.charge:.0%}" if effects_on else "EFFECTS OFF"
        if audio.muted:
            status += "  [SFX MUTED]"
        if vanish.mix >= 0.5:
            status += "  [INVISIBLE]"
        cv2.putText(preview, status, (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if lm is not None:
            cv2.putText(preview, f"openness {classifier.openness:.2f}  palm {facing(normal)}", (16, 142),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, f"fingers {classifier.fingers}  thumb {classifier.thumb:+.2f}", (16, 172),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        message = vanish.message(now) or ("" if vanish.plate is not None else "B: capture the empty room for the vanishing jutsu")
        if message:
            cv2.putText(preview, message, (16, preview.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow("Hand VFX Camera (Space effects, M mute, B background, D skeleton, Q quit)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            effects_on = not effects_on
        if key == ord("d"):
            show_skeleton = not show_skeleton
        if key == ord("m"):
            audio.muted = not audio.muted  # SFX only, for you and the meeting; your voice keeps going
        if key == ord("b"):
            vanish.start_capture(now)

    if vcam is not None:
        vcam.close()
    audio.close()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
