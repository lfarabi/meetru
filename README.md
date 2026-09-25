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

Gesture: buka telapak → orb muncul · tahan orb dengan telapak ke atas/samping → charging (maks 3 detik, suara makin tinggi) · arahkan telapak ke kamera (setelah charge ≥0.5 detik) → blast · kepal atau tangan keluar frame → orb padam. Orb mengikuti arah hadap telapak (3D): telapak ke atas → orb di atas tangan; punggung tangan ke kamera → orb tertutup tangan. Preview menampilkan `palm CAMERA/UP/DOWN/LEFT/RIGHT/AWAY`.

Kamera dipilih lewat `camera.device_id` di `config.yaml`. Webcam hanya bisa dipakai satu aplikasi: tutup NVIDIA Broadcast / Meet / Zoom yang memegang kamera fisik.

SFX keluar ke output default Windows; ganti lewat `audio.output_device` (sebagian nama perangkat, mis. `"USB Audio CODEC"`).

### Gesture jari → sound effect

| Gesture (jempol dilipat) | Suara bawaan | Mode |
|---|---|---|
| ☝️ 1 jari | klik / beep | sekali |
| ✌️ 2 jari | swoosh | sekali |
| 3 jari | synth naik + ledakan | sekali |
| 4 jari | beat ritmis | loop selama ditahan |

Mengganti suara: taruh file MP3/WAV/OGG/FLAC di `assets/audio/` (mis. `airhorn.mp3`), lalu di `config.yaml` (nama tanpa ekstensi):

```yaml
finger_sfx:
  TWO: {sound: airhorn, mode: once, volume: 1.0}
```

Semua file media di `assets/audio/` dan `assets/video/` tidak ikut di-commit (`.gitignore`); di PC lain salin ulang file-nya. Suara berbunyi selama gesture jarinya ditahan dan berhenti saat dilepas (maksimal sepanjang file-nya). Hanya satu sound effect berbunyi pada satu waktu: ganti jari (atau buka telapak / blast) langsung menghentikan suara sebelumnya. Gesture jari harus ditahan ±0.15 detik (`finger_sfx.hold`) sebelum berbunyi, dan tidak berbunyi saat orb sedang charge/blast.

Preview menampilkan `fingers N thumb ±x.xx` untuk tuning `gesture.finger_straight` dan `gesture.thumb_out`.

### Orb video
Orb memakai `assets/video/orb.mp4` (`orb.video` di `config.yaml`): klip berlatar hitam dengan orb di tengah, ditempel di telapak dengan blending screen. Awal klip (orb terbentuk) diputar saat orb muncul, sisanya di-loop; ukuran bola dideteksi otomatis (`orb.video_scale` untuk menyesuaikan). Ganti klipnya cukup dengan menaruh video lain dan mengubah `orb.video`. Kalau file tidak ada, orb prosedural yang dipakai.

## Pakai di Google Meet / Zoom
Sekali saja: pasang **VB-CABLE** (https://vb-audio.com/Cable/ → unduh zip → klik kanan `VBCABLE_Setup_x64.exe` → Run as administrator → Install Driver → restart). Setelah itu pastikan output default Windows tetap speaker/headphone Anda, bukan "CABLE Input".

1. Tutup NVIDIA Broadcast (dan app lain yang memegang webcam).
2. Jalankan `run.bat` **dulu**. Baris atas preview harus `VCAM ON  MEET AUDIO ON`.
3. Di Meet: ⋮ → Settings →
   - **Video** → Camera: **OBS Virtual Camera** (jangan "FHD Camera" — webcam sedang dipakai aplikasi ini)
   - **Audio** → Microphone: **CABLE Output (VB-Audio Virtual Cable)**; kalau SFX terpotong/hilang, matikan *Noise cancellation*
4. Pakai **headphone**: SFX juga diputar ke speaker Anda, dan kalau lewat speaker, mic akan menangkapnya lagi (gema).

Suara Anda tetap lewat (mic USB Audio CODEC dicampur di aplikasi; ganti lewat `meeting_audio.mic_device`, mis. `FHD Camera` untuk mic webcam); `M` hanya membisukan SFX, bukan suara Anda. Untuk mute suara, pakai tombol mute di Meet.

Meet me-mirror tampilan diri Anda sendiri; peserta lain melihat versi tidak ter-mirror (tulisan terbaca normal). Overlay debug (FPS, gesture) hanya ada di preview, tidak terkirim. Aplikasi OBS Studio tidak perlu dibuka; kalau OBS terbuka dengan "Start Virtual Camera" aktif, matikan dulu (preview akan menulis `VCAM OFF`).

## Tes
```bash
.venv\Scripts\python -m pytest -q
```
