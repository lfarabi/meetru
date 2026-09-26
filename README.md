# Hand VFX Camera

Webcam hand tracking → gesture → VFX/SFX, keluaran ke OBS Virtual Camera. Keputusan desain: [DECISIONS.md](DECISIONS.md).

## Setup (sekali)
```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o models/selfie_multiclass_256x256.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

## Jalankan
Klik dua kali `run.bat`, atau `.venv\Scripts\python main.py`.

| Tombol (di jendela preview) | Fungsi |
|---|---|
| `Space` | efek on/off |
| `M` | mute SFX |
| `B` | rekam ruangan kosong untuk jutsu menghilang (hitung mundur 5 detik, keluar dari frame) |
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

Semua file media di `assets/audio/` dan `assets/video/` tidak ikut di-commit (`.gitignore`); di PC lain salin ulang file-nya. Suara berbunyi selama gesture jarinya ditahan dan berhenti saat dilepas (maksimal sepanjang file-nya). Hanya satu sound effect berbunyi pada satu waktu: ganti jari (atau buka telapak / blast) langsung menghentikan suara sebelumnya. Gesture jari harus ditahan ±0,3 detik (`finger_sfx.hold`) sebelum berbunyi, dan tidak berbunyi saat orb sedang charge/blast atau saat pose dua tangan (segel, wow) sedang dibentuk: sampai 0,6 detik setelah dua tangan terakhir terlihat (`gesture.two_hand_grace`), satu tangan dianggap bagian dari pose, dan suara jari yang terlanjur berbunyi dihentikan.

Preview menampilkan `fingers N thumb ±x.xx` untuk tuning `gesture.finger_straight` dan `gesture.thumb_out`.

### Orb video
Orb memakai `assets/video/orb.mp4` (`orb.video` di `config.yaml`): klip berlatar hitam dengan orb di tengah, ditempel di telapak dengan blending screen. Awal klip (orb terbentuk) diputar saat orb muncul, sisanya di-loop; ukuran bola dideteksi otomatis (`orb.video_scale` untuk menyesuaikan). Ganti klipnya cukup dengan menaruh video lain dan mengubah `orb.video`. Kalau file tidak ada, orb prosedural yang dipakai.

### Jutsu menghilang (seal)
1. Sekali saja (dan setiap kali posisi kamera/cahaya berubah): tekan `B`, keluar dari frame dalam 5 detik, tunggu tulisan *Empty room captured*. Foto ruangan kosong disimpan di `assets/background.png` (tidak di-commit).
2. Buat segel: dua jari (telunjuk + tengah) di kedua tangan, satu tegak dan satu mendatar, disilangkan; tahan ±0,3 detik → suara `seal_sound.mp3` + asap `assets/video/smoke.mp4` menyembur di tempat segel dan Anda menghilang (tinggal ruangan kosong).
3. Segel lagi → asap lagi → Anda muncul kembali.

Asap green screen di-key otomatis; bagian terpadatnya diletakkan di titik segel (`vanish.smoke_height` untuk ukurannya). Suara jutsu tidak bisa ditimpa: selama ±3,6 detik itu suara lain (orb, blast, jari, dengung charge) diabaikan, dan selama asap orb tidak bisa dimunculkan. Setelah itu orb tetap bisa dipakai saat tak terlihat. SFX jari dibisukan saat dua tangan terlihat supaya segel tidak memicu suara 2 jari.

### Pose wow
Kedua telapak terbuka, pergelangan berdempetan di bawah dagu, jari ke atas (seperti membingkai wajah); tahan ±0,3 detik → suara `anime-wow-sound-effect.mp3` + klip green screen `assets/video/wow.mp4` muncul layar penuh sesuai framing aslinya. Suaranya eksklusif seperti suara jutsu. Jangan terlalu merapatkan kedua tangan: kalau bertumpuk, MediaPipe bisa hanya melihat satu tangan.

### Kepalan terangkat → Raised-fist sound
Angkat kepalan (pergelangan di 60% atas frame), tahan ±0,3 detik → suara `raised-fist-sound.mp3` + Figure muncul di pojok bawah (kiri di preview Anda, kanan bagi peserta), tinggi 80% frame, selama kepalan tetap terangkat (maks 9,5 detik); turunkan kepalan → Figure memudar. Klipnya `assets/video/surprise.mp4` (green screen, di-key otomatis). Untuk video biasa tanpa green screen, tambahkan `segmenter: models/selfie_multiclass_256x256.tflite` di bagian `figure` pada `config.yaml`: latarnya dihapus dengan AI segmentasi (orang utama = wajah terbesar), pertama kali ±20 detik lalu di-cache (`*.cutout.npz`).

Dua tangan terlihat = mode pose: orb tidak muncul/charge dan SFX jari dibisukan.

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
