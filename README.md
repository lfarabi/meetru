# Hand VFX Camera

Webcam hand tracking → gesture → VFX/SFX, keluaran ke OBS Virtual Camera. Keputusan desain: [DECISIONS.md](DECISIONS.md).

## Setup (sekali)
```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Jalankan
Klik dua kali `run.bat`, atau `.venv\Scripts\python main.py`. `Q`/`Esc` untuk keluar.

Kamera dipilih lewat `camera.device_id` di `config.yaml`. Webcam hanya bisa dipakai satu aplikasi: tutup NVIDIA Broadcast / Meet / Zoom yang memegang kamera fisik.

## Tes
```bash
.venv\Scripts\python -m pytest -q
```
