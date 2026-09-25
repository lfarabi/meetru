# Hand VFX Camera

Webcam hand tracking → gesture → VFX/SFX, keluaran ke OBS Virtual Camera. Keputusan desain: [DECISIONS.md](DECISIONS.md).

## Setup (sekali)
```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Jalankan
Klik dua kali `run.bat`, atau `.venv\Scripts\python main.py`.

| Tombol (di jendela preview) | Fungsi |
|---|---|
| `Space` | efek on/off |
| `M` | mute SFX |
| `D` | skeleton tangan (debug) |
| `Q` / `Esc` | keluar |

Gesture: buka telapak → orb muncul · kepal → charging (maks 3 detik, suara makin tinggi) · kepal ≥0.5 detik lalu buka cepat → blast · tangan keluar frame → orb padam. Orb mengikuti arah hadap telapak (3D): telapak ke atas → orb di atas tangan; punggung tangan ke kamera → orb tertutup tangan. Preview menampilkan `palm CAMERA/UP/DOWN/LEFT/RIGHT/AWAY`.

Kamera dipilih lewat `camera.device_id` di `config.yaml`. Webcam hanya bisa dipakai satu aplikasi: tutup NVIDIA Broadcast / Meet / Zoom yang memegang kamera fisik.

SFX keluar ke output default Windows; ganti lewat `audio.output_device` (sebagian nama perangkat, mis. `"USB Audio CODEC"`).

### Gesture jari → sound effect

| Gesture (jempol dilipat) | Suara bawaan | Mode |
|---|---|---|
| ☝️ 1 jari | klik / beep | sekali |
| ✌️ 2 jari | swoosh | sekali |
| 3 jari | synth naik + ledakan | sekali |
| 4 jari | beat ritmis | loop selama ditahan |

Mengganti suara: taruh WAV 16-bit di `assets/audio/` (mis. `airhorn.wav`), lalu di `config.yaml`:

```yaml
finger_sfx:
  TWO: {sound: airhorn, mode: once, volume: 1.0}
```

Preview menampilkan `fingers N thumb ±x.xx` untuk tuning `gesture.finger_straight` dan `gesture.thumb_out`.

## Tes
```bash
.venv\Scripts\python -m pytest -q
```
