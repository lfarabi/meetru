"""Gesture SFX: placeholder sounds synthesized with NumPy (overridable by WAVs) and a low-latency mixer.

Audio follows orb *state* transitions, never the renderer, so it stays independent of the VFX.
"""
import logging
import queue
import wave
from pathlib import Path

import numpy as np

from gestures import ACTIVE, BLAST, CHARGING, FADING, HIDDEN

log = logging.getLogger(__name__)
SOUNDS = ("activation", "charge_loop", "blast", "power_down")
ASSET_DIR = Path("assets/audio")


def _band(x: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    """Brick-wall band-pass via FFT; fine for offline synthesis."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


def _chirp(f0: float, f1: float, seconds: float, sr: int) -> np.ndarray:
    """Exponential sine sweep from f0 to f1 Hz."""
    t = np.arange(int(seconds * sr)) / sr
    k = f1 / f0
    return np.sin(2 * np.pi * f0 * seconds / np.log(k) * (k ** (t / seconds) - 1))


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

    return {"activation": activation, "charge_loop": charge_loop, "blast": blast, "power_down": power_down}


def load_wav(path: Path, sr: int) -> np.ndarray:
    """Load a 16-bit PCM WAV as mono float32 at `sr` (linear resampling if needed)."""
    with wave.open(str(path)) as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        data = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    x = np.frombuffer(data, np.int16).astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768
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


class Mixer:
    """One-shot voices plus a pitch-shiftable charge loop. `render` runs on the audio thread."""

    def __init__(self, sounds: dict[str, np.ndarray], cfg: dict):
        self.sounds, self.cfg = sounds, cfg
        self.muted = False
        self.loop_gain_target, self.loop_rate_target = 0.0, 1.0  # written by the main thread
        self._queue = queue.SimpleQueue()
        self._voices = []  # [buffer, position, gain]
        self._loop_gain, self._loop_rate, self._loop_pos = 0.0, 1.0, 0.0

    def play(self, name: str, gain: float = 1.0):
        self._queue.put((name, gain))

    def render(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        while not self._queue.empty():
            name, gain = self._queue.get_nowait()
            self._voices.append([self.sounds[name], 0, gain * self.cfg["sfx_volume"]])
        for v in self._voices:
            chunk = v[0][v[1]:v[1] + n]
            out[:len(chunk)] += chunk * v[2]
            v[1] += n
        self._voices = [v for v in self._voices if v[1] < len(v[0])]

        target = self.loop_gain_target * self.cfg["charge_volume"]
        if target > 0 or self._loop_gain > 1e-4:
            loop = self.sounds["charge_loop"]
            self._loop_rate += (self.loop_rate_target - self._loop_rate) * 0.3
            idx = self._loop_pos + np.arange(n) * self._loop_rate  # resampling = pitch shift
            i0 = idx.astype(np.int64) % len(loop)
            frac = (idx - np.floor(idx)).astype(np.float32)
            sample = loop[i0] * (1 - frac) + loop[(i0 + 1) % len(loop)] * frac
            new_gain = self._loop_gain + (target - self._loop_gain) * min(1.0, n / (0.06 * self.cfg["sample_rate"]))
            out += sample * np.linspace(self._loop_gain, new_gain, n, dtype=np.float32)  # click-free ramp
            self._loop_gain = new_gain
            self._loop_pos = (self._loop_pos + n * self._loop_rate) % len(loop)

        return np.tanh(out * (0.0 if self.muted else self.cfg["master_volume"]))  # soft clip


class AudioEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        sounds = synthesize(cfg["sample_rate"])
        for name in SOUNDS:
            path = ASSET_DIR / f"{name}.wav"
            if path.exists():
                try:
                    sounds[name] = load_wav(path, cfg["sample_rate"])
                    log.info("Using %s", path)
                except Exception as e:
                    log.warning("Ignoring %s: %s", path, e)
        self.mixer = Mixer(sounds, cfg)
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

    def update(self, prev: str, state: str, charge: float, blast_power: float):
        """Map orb state transitions to SFX; call once per frame."""
        if state != prev:
            if state == ACTIVE and prev == HIDDEN:
                self.mixer.play("activation")
            elif state == BLAST:
                self.mixer.play("blast", 0.6 + 0.4 * blast_power)
            elif state == FADING:
                self.mixer.play("power_down", 0.7)
        charging = state == CHARGING
        self.mixer.loop_gain_target = 0.5 + 0.5 * charge if charging else 0.0
        self.mixer.loop_rate_target = 1.0 + 0.8 * charge  # pitch rises with charge

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
