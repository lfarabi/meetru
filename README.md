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

Gesture: buka telapak → orb muncul · kepal → charging (maks 3 detik, suara makin tinggi) · kepal ≥0.5 detik lalu buka cepat → blast · tangan keluar frame → orb padam. Orb mengikuti arah hadap telapak (3D): telapak ke atas → orb di atas tangan; punggung tangan ke kamera → orb tertutup tangan. Preview menampilkan `palm CAMERA/UP/DOWN/LEFT/RIGHT/AWAY`; kalau arahnya terbalik, set `orb.normal_flip: true`.

Kamera dipilih lewat `camera.device_id` di `config.yaml`. Webcam hanya bisa dipakai satu aplikasi: tutup NVIDIA Broadcast / Meet / Zoom yang memegang kamera fisik.

SFX keluar ke output default Windows; ganti lewat `audio.output_device` (sebagian nama perangkat, mis. `"USB Audio CODEC"`). SFX sendiri bisa ditaruh di `assets/audio/`.

## Tes
```bash
.venv\Scripts\python -m pytest -q
```
