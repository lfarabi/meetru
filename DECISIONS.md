# Keputusan Proyek — Hand VFX Camera

Hasil sesi grilling 2026-09-25 atas `hand-vfx-camera-project-handoff.md`. Jika bertentangan dengan handoff, dokumen ini yang berlaku.

## Target
- Pemakaian pribadi di PC ini, dijalankan via `run.bat`. Tidak ada packaging `.exe`.
- Python 3.12 di `.venv` (MediaPipe belum mendukung 3.14 yang jadi default sistem).
- Target meeting pertama: Google Meet di Chrome.

## Input & tracking
- Kamera: "FHD Camera" = DirectShow index 0 (saat NVIDIA Broadcast tertutup), 1280×720 MJPG. Terukur ±25 FPS (dibatasi kamera/pencahayaan; tracking hanya ±11 ms/frame). NVIDIA Broadcast harus ditutup karena memegang webcam.
- Satu tangan saja pada MVP.
- Gesture: NO_HAND / OPEN_PALM / FIST. Jarak fingertip dinormalisasi dengan ukuran telapak (wrist→middle MCP), ada hysteresis + debounce agar tidak flicker.

## Efek
- Energy Orb biru-cyan, inti putih: spawn, mengikuti tangan, charge hingga 3 detik.
- **Blast = lepas kepalan** (bukan dorongan/z-depth): charge ≥0.5 s lalu buka <250 ms → ledakan, kekuatan ∝ lama charge, cooldown 1 s. Charge <0.5 s → kembali ACTIVE.
- Semua angka di `config.yaml`.

## Output
- Video: `pyvirtualcam` langsung ke driver OBS Virtual Camera (aplikasi OBS tidak perlu dibuka).
- Video ke meeting **tidak** di-mirror; preview lokal di-mirror + overlay debug (FPS, gesture, state). Overlay tidak masuk ke meeting.
- Audio: SFX placeholder disintesis NumPy. Mixer `sounddevice`: mic **USB Audio CODEC** + SFX → VB-CABLE Input → mic "CABLE Output" di Meet. SFX juga dimonitor ke headphone (bisa dimatikan).
- VB-CABLE diinstal manual oleh user saat Checkpoint 4. Jika Meet memotong SFX, matikan noise cancellation Meet.

## Kontrol (jendela preview)
- `Space` efek on/off · `M` mute SFX · `Q`/`Esc` keluar.

## Struktur kode
Ramping: `main.py`, `tracking.py`, `gestures.py`, `orb.py`, `audio.py`, `config.yaml`, `tests/`. Pecah file hanya saat efek kedua muncul. Repo: github.com/lfarabi/meetcam (private, GPL-3.0), branch `main`.

## Checkpoint (user menguji dengan tangan sendiri)
1. Preview + skeleton + label gesture + FPS.
2. Orb + charging + blast.
3. Virtual camera di Google Meet.
4. VB-CABLE + SFX + mic di Meet.
