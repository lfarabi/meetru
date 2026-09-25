"""Gesture SFX: placeholder sounds synthesized with NumPy (overridable by WAVs) and a low-latency mixer.

Audio follows orb *state* transitions and finger-count gestures, never the renderer, so it stays
independent of the VFX.
"""
import logging
import queue
from pathlib import Path

import numpy as np
import soundfile as sf

from gestures import ACTIVE, BLAST, CHARGING, FADING, FINGER_GESTURES, HIDDEN

log = logging.getLogger(__name__)
ORB_SOUNDS = ("activation", "charge_loop", "blast", "power_down")
ASSET_DIR = Path("assets/audio")
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac")


def _band(x: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    """Brick-wall band-pass via FFT; fine for offline synthesis."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


def _sweep_phase(f0: float, f1: float, seconds: float, sr: int) -> np.ndarray:
    """Phase (radians) of an exponential sweep from f0 to f1 Hz."""
    t = np.arange(int(seconds * sr)) / sr
    k = f1 / f0
    return 2 * np.pi * f0 * seconds / np.log(k) * (k ** (t / seconds) - 1)


def _chirp(f0: float, f1: float, seconds: float, sr: int) -> np.ndarray:
    return np.sin(_sweep_phase(f0, f1, seconds, sr))


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    return (x * peak / max(np.abs(x).max(), 1e-9)).astype(np.float32)


def synthesize(sr: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)

    def t(seconds):
        return np.arange(int(seconds * sr)) / sr

    # activation: rising whoosh + shimmer
    d = 0.6
    env = np.sin(np.pi * np.clip(t(d) / d, 0, 1)) ** 2
    whoosh = _band(rng.standard_normal(len(env)), 600, 5000, sr) * env
    rise = _chirp(220, 880, d, sr) * np.exp(-3 * t(d)) * np.minimum(1, t(d) / 0.02)
    activation = _norm(whoosh * 0.6 + rise * 0.5 + _chirp(1800, 2600, d, sr) * env * 0.1)

    # charge loop: 2 s hum; integer Hz over 2 s and FFT noise are periodic, so it loops seamlessly
    d = 2.0
    tt = t(d)
    hum = sum(a * np.sin(2 * np.pi * f * tt) for f, a in ((110, 1.0), (220, 0.5), (331, 0.3), (440, 0.15)))
    buzz = _band(rng.standard_normal(len(tt)), 150, 1200, sr) * 0.4
    charge_loop = _norm((hum + buzz) * (0.8 + 0.2 * np.sin(2 * np.pi * 6 * tt)), 0.7)

    # blast: sharp crack, noise burst, sub-bass drop
    d = 1.3
    tt = t(d)
    burst = _band(rng.standard_normal(len(tt)), 40, 2500, sr) * np.exp(-tt / 0.25)
    crack = _band(rng.standard_normal(len(tt)), 2000, 9000, sr) * np.exp(-tt / 0.03)
    sub = _chirp(140, 35, d, sr) * np.exp(-tt / 0.5)
    blast = _norm(np.tanh((burst * 0.8 + crack * 0.6 + sub * 1.2) * 1.5) * np.minimum(1, tt / 0.003))

    # power down: falling tone with vibrato, fading out
    d = 0.7
    tt = t(d)
    fall = _chirp(700, 90, d, sr) * (1 + 0.1 * np.sin(2 * np.pi * 9 * tt))
    power_down = _norm(fall * (1 - tt / d) ** 2 + _band(rng.standard_normal(len(tt)), 100, 800, sr) * 0.2 * (1 - tt / d))

    # finger 1: short click / beep
    d = 0.18
    tt = t(d)
    finger_1 = _norm(np.sin(2 * np.pi * 1320 * tt) * np.exp(-tt / 0.05)
                     + rng.standard_normal(len(tt)) * np.exp(-tt / 0.002) * 0.5)

    # finger 2: swoosh, noise sweeping from a low band to a high band
    d = 0.8
    tt = t(d)
    x = tt / d
    swoosh = (_band(rng.standard_normal(len(tt)), 300, 2000, sr) * (1 - x)
              + _band(rng.standard_normal(len(tt)), 1500, 8000, sr) * x)
    finger_2 = _norm(swoosh * np.sin(np.pi * x) ** 1.5)

    # finger 3: synth rise, then explosion
    rise_d, boom_d = 1.1, 1.2
    phase = _sweep_phase(110, 880, rise_d, sr)
    saw = sum(np.sin(k * phase) / k for k in range(1, 9)) * (t(rise_d) / rise_d) ** 2
    tb = t(boom_d)
    boom = np.tanh((_band(rng.standard_normal(len(tb)), 40, 3000, sr) * np.exp(-tb / 0.3)
                    + _chirp(120, 35, boom_d, sr) * np.exp(-tb / 0.45) * 1.2) * 1.6)
    finger_3 = _norm(np.concatenate([saw / np.abs(saw).max() * 0.6, boom]))

    # finger 4: 2 s beat loop at 120 BPM (kick, snare, hats); every hit decays before the loop point
    d = 2.0
    beat = np.zeros(int(d * sr))

    def hit(start, sound):
        i = int(start * sr)
        beat[i:i + len(sound)] += sound[:len(beat) - i]

    tk = t(0.25)
    for start in (0.0, 1.0):
        hit(start, np.sin(_sweep_phase(150, 45, 0.25, sr)) * np.exp(-tk / 0.12))
    for start in (0.5, 1.5):
        hit(start, _band(rng.standard_normal(len(tk)), 1000, 6000, sr) * np.exp(-tk / 0.06) * 0.8
            + np.sin(2 * np.pi * 200 * tk) * np.exp(-tk / 0.05) * 0.4)
    th = t(0.1)
    for start in np.arange(0, d, 0.25):
        hit(start, _band(rng.standard_normal(len(th)), 7000, 12000, sr) * np.exp(-th / 0.02) * 0.3)
    finger_4 = _norm(beat, 0.8)

    return {"activation": activation, "charge_loop": charge_loop, "blast": blast, "power_down": power_down,
            "finger_1": finger_1, "finger_2": finger_2, "finger_3": finger_3, "finger_4": finger_4}


def find_asset(name: str) -> Path | None:
    """assets/audio/<name> with the first supported extension that exists."""
    return next((p for ext in AUDIO_EXTS if (p := ASSET_DIR / f"{name}{ext}").exists()), None)


def load_sound(path: Path, sr: int) -> np.ndarray:
    """Load WAV/MP3/OGG/FLAC as mono float32 at `sr` (linear resampling if needed)."""
    x, rate = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if rate != sr:
        x = np.interp(np.arange(0, len(x), rate / sr), np.arange(len(x)), x).astype(np.float32)
    return x


def find_device(sd, host_api: str, name: str | None, kind: str) -> int:
    """Device index on `host_api` whose name contains `name`, or that API's default device if name is None."""
    api = next(i for i, h in enumerate(sd.query_hostapis()) if h["name"] == host_api)
    if name is None:
        return sd.query_hostapis(api)[f"default_{kind}_device"]
    channels = f"max_{kind}_channels"
    for i, dev in enumerate(sd.query_devices()):
        if dev["hostapi"] == api and dev[channels] > 0 and name.lower() in dev["name"].lower():
            return i
    raise LookupError(f"no {kind} device matching {name!r} on {host_api}")


class _Loop:
    def __init__(self, buffer: np.ndarray):
        self.buffer = buffer
        self.target_gain, self.target_rate = 0.0, 1.0  # written by the main thread
        self.gain, self.rate, self.pos = 0.0, 1.0, 0.0  # owned by the audio thread


class Mixer:
    """One SFX channel (a new one-shot cuts the previous one) plus loops with gain/pitch control.

    `render` runs on the audio thread.
    """

    def __init__(self, sounds: dict[str, np.ndarray], cfg: dict, loops: tuple[str, ...] = ("charge_loop",)):
        self.sounds, self.cfg = sounds, cfg
        self.muted = False
        self._queue = queue.SimpleQueue()
        self._voices = []  # [name, buffer, position, gain, fading_out]; only the newest is not fading
        # created up front so the audio thread never sees the dict change size
        self._loops = {name: _Loop(sounds[name]) for name in loops}

    def play(self, name: str, gain: float = 1.0):
        """Start a one-shot, fading out whatever one-shot is still playing so sounds never pile up."""
        self._queue.put((name, gain))

    def set_loop(self, name: str, gain: float, rate: float = 1.0):
        """Fade a loop toward `gain` (0 = stop) at playback `rate` (pitch)."""
        loop = self._loops[name]
        loop.target_gain, loop.target_rate = gain, rate

    def render(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        while not self._queue.empty():
            name, gain = self._queue.get_nowait()
            for v in self._voices:
                v[4] = True
            self._voices.append([name, self.sounds[name], 0, gain * self.cfg["sfx_volume"], False])
        for v in self._voices:
            _, buf, pos, gain, fading = v
            chunk = buf[pos:pos + n] * gain
            if fading:  # one-block fade instead of a hard cut, which would click
                chunk *= np.linspace(1, 0, len(chunk), dtype=np.float32)
                v[2] = len(buf)
            out[:len(chunk)] += chunk
            v[2] += n
        self._voices = [v for v in self._voices if v[2] < len(v[1])]

        for lp in self._loops.values():
            if lp.target_gain <= 0 and lp.gain <= 1e-4:
                continue
            buf = lp.buffer
            lp.rate += (lp.target_rate - lp.rate) * 0.3
            idx = lp.pos + np.arange(n) * lp.rate  # resampling = pitch shift
            i0 = idx.astype(np.int64) % len(buf)
            frac = (idx - np.floor(idx)).astype(np.float32)
            sample = buf[i0] * (1 - frac) + buf[(i0 + 1) % len(buf)] * frac
            new_gain = lp.gain + (lp.target_gain - lp.gain) * min(1.0, n / (0.06 * self.cfg["sample_rate"]))
            out += sample * np.linspace(lp.gain, new_gain, n, dtype=np.float32)  # click-free ramp
            lp.gain = new_gain
            lp.pos = (lp.pos + n * lp.rate) % len(buf)

        return np.tanh(out * (0.0 if self.muted else self.cfg["master_volume"]))  # soft clip


class AudioEngine:
    def __init__(self, cfg: dict, finger_cfg: dict):
        self.cfg = cfg
        self.cooldown, self.hold = finger_cfg["cooldown"], finger_cfg["hold"]
        sounds = synthesize(cfg["sample_rate"])
        wanted = set(ORB_SOUNDS) | {finger_cfg[g]["sound"] for g in FINGER_GESTURES if g in finger_cfg}
        for name in wanted:
            path = find_asset(name)
            if path:
                try:
                    sounds[name] = load_sound(path, cfg["sample_rate"])
                    log.info("Using %s", path)
                except Exception as e:
                    log.warning("Ignoring %s: %s", path, e)
        # gesture -> {sound, mode, volume}; a mapping whose sound can't be found is dropped with a warning
        self.fingers = {}
        for g in FINGER_GESTURES:
            spec = finger_cfg.get(g)
            if spec and spec["sound"] in sounds:
                self.fingers[g] = spec
            elif spec:
                log.warning("No sound %r for %s: put %s.mp3 (or .wav/.ogg/.flac) in %s",
                            spec["sound"], g, spec["sound"], ASSET_DIR)
        loops = ("charge_loop",) + tuple({s["sound"] for s in self.fingers.values() if s["mode"] == "loop"})
        self.mixer = Mixer(sounds, cfg, loops)
        self._last_played = {}
        self._gesture, self._since, self._fired = None, 0.0, False  # finger gesture being held
        self._stream = None
        if not cfg["enabled"]:
            return
        try:
            import sounddevice as sd
            device = find_device(sd, cfg["host_api"], cfg["output_device"], "output")
            wasapi = sd.WasapiSettings(auto_convert=True) if "WASAPI" in cfg["host_api"] else None
            self._stream = sd.OutputStream(samplerate=cfg["sample_rate"], channels=2, dtype="float32",
                                           blocksize=cfg["blocksize"], device=device, latency="low",
                                           extra_settings=wasapi, callback=self._callback)
            self._stream.start()
            log.info("Audio out: %s (latency %.0f ms)", sd.query_devices(self._stream.device)["name"],
                     self._stream.latency * 1000)
        except Exception as e:
            log.warning("Audio disabled, output device failed: %s", e)
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        outdata[:] = self.mixer.render(frames)[:, None]

    def update(self, prev: str, state: str, charge: float, blast_power: float, silent_fade: bool = False):
        """Map orb state transitions to SFX; call once per frame."""
        if state != prev:
            if state == ACTIVE and prev == HIDDEN:
                self.mixer.play("activation")
            elif state == BLAST:
                self.mixer.play("blast", 0.6 + 0.4 * blast_power)
            elif state == FADING and not silent_fade:
                self.mixer.play("power_down", 0.7)
        gain = (0.5 + 0.5 * charge) * self.cfg["charge_volume"] if state == CHARGING else 0.0
        self.mixer.set_loop("charge_loop", gain, 1.0 + 0.8 * charge)  # pitch rises with charge

    def update_fingers(self, gesture: str, t: float, orb_busy: bool = False):
        """Finger-count SFX; call once per frame.

        A finger gesture must be held for `hold` seconds first, so the counts the hand passes through
        while opening or closing (e.g. releasing a fist into a blast) don't fire. Nothing fires while
        the orb is charging or blasting. 'once' plays when the hold is reached (rate-limited per
        gesture); 'loop' plays for as long as the gesture is held.
        """
        if gesture != self._gesture:
            self._gesture, self._since, self._fired = gesture, t, False
        settled = t - self._since >= self.hold and not orb_busy
        for g, spec in self.fingers.items():
            if spec["mode"] == "loop":
                self.mixer.set_loop(spec["sound"], spec["volume"] if gesture == g and settled else 0.0)
        spec = self.fingers.get(gesture)
        if spec and spec["mode"] == "once" and settled and not self._fired:
            self._fired = True
            if t - self._last_played.get(gesture, float("-inf")) >= self.cooldown:
                self.mixer.play(spec["sound"], spec["volume"])
                self._last_played[gesture] = t

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
