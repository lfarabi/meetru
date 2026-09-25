# Keputusan Proyek — Hand VFX Camera

Hasil sesi grilling 2026-09-25 atas `hand-vfx-camera-project-handoff.md`. Jika bertentangan dengan handoff, dokumen ini yang berlaku.

## Target
- Pemakaian pribadi di PC ini, dijalankan via `run.bat`. Tidak ada packaging `.exe`.
- Python 3.12 di `.venv` (MediaPipe belum mendukung 3.14 yang jadi default sistem).
- Target meeting pertama: Google Meet di Chrome.

## Input & tracking
- Kamera: "FHD Camera" = DirectShow index 0 (saat NVIDIA Broadcast tertutup), 1280×720 MJPG. Terukur ±25 FPS (dibatasi kamera/pencahayaan; tracking hanya ±11 ms/frame). NVIDIA Broadcast harus ditutup karena memegang webcam.
- Satu tangan saja pada MVP.
- Gesture: NO_HAND / OPEN_PALM / FIST / ONE / TWO / THREE / FOUR, dengan debounce agar tidak flicker.
- Jari terangkat dihitung dari landmark 3D (kelurusan jari > 0.82 dan ujung melewati sendi tengah) → tahan rotasi tangan. Jempol "keluar" = ujung jempol di luar tepi telapak sisi telunjuk (> 0.1 panjang telapak). Angka awal diukur dari gambar referensi user (jari lurus ≥ 0.94, terlipat ≤ 0.71; jempol keluar +0.3…+0.56, terlipat −0.2…−0.4).
- Prioritas: ≥3 jari + jempol keluar → OPEN_PALM; 1–4 jari → ONE…FOUR (dihitung jumlahnya, jari mana pun; jempol diabaikan kecuali membedakan FOUR vs OPEN_PALM); 0 jari + openness < 0.55 → FIST. Jari terangkat menang atas openness, karena 1 jari hanya sedikit menaikkan openness rata-rata (0.59 di gambar referensi, nyaris terbaca FIST).

## Efek
- Energy Orb gaya Orb (sesuai gambar referensi user, dibuat prosedural karena PNG referensi berlatar checkerboard): bola biru pekat semi-solid, garis cahaya kusut berputar, inti putih, 5 sabetan sabit biru di orbit miring, percikan. Charge hingga 3 detik.
- Posisi: **tengah telapak** = titik tengah antara wrist dan rata-rata 4 MCP (`orb.palm_position: 0.5`). Rata-rata 5 titik dari handoff terlalu condong ke buku jari.
- **Orb 3D**: orb melayang dari tengah telapak searah normal telapak (dari world landmarks MediaPipe, dismoothing), menempel di permukaan (`hover: 1.0` radius). Telapak ke atas → orb di atas tangan; ke bawah → menggantung di bawah (tidak hilang).
- Telapak ±menghadap kamera (|normal z| > `center_lock` 0.75) → orb dikunci di tengah telapak, agar kemiringan kecil / noise kedalaman tidak menggeser orb.
- Punggung tangan ke kamera (normal z > 0.65) → orb di balik tangan: siluet tangan dari landmark menutupi orb (oklusi, ±1.5 ms). Telapak ke atas yang sedikit condong tidak memicu oklusi.
- **Sisi telapak ditentukan dari lengkung jari**, bukan label kiri/kanan MediaPipe: diuji pada screenshot user, label handedness MediaPipe Tasks tertukar pada pose miring (telapak ke atas), sedangkan sisi lengkung jari/jempol benar di semua kasus (selisih 7–31 mm). Tangan hampir datar mempertahankan sisi sebelumnya.
- **Blast = lepas kepalan** (bukan dorongan/z-depth): charge ≥0.5 s lalu buka <250 ms → ledakan, kekuatan ∝ lama charge, cooldown 1 s. Charge <0.5 s → kembali ACTIVE.
- Semua angka di `config.yaml`.

## Output
- Video: `pyvirtualcam` langsung ke driver OBS Virtual Camera (aplikasi OBS tidak perlu dibuka). Diverifikasi ujung-ke-ujung: frame dibaca kembali dari "OBS Virtual Camera" (DirectShow index 2) cocok dengan yang dikirim (selisih rata-rata 5.3, tidak ter-mirror, orb ikut, overlay tidak). Driver sibuk / tidak ada → aplikasi tetap jalan preview-only (`VCAM OFF`).
- Video ke meeting **tidak** di-mirror; preview lokal di-mirror + overlay debug (FPS, gesture, state). Overlay tidak masuk ke meeting.
- Audio: SFX placeholder disintesis NumPy (bisa diganti file MP3/WAV/OGG/FLAC di `assets/audio/`). SFX lokal dimajukan ke Checkpoint 2 atas permintaan user: WASAPI (±22 ms; MME ±96 ms) ke output default Windows (saat ini Speakers Realtek), bisa diganti via `audio.output_device`.
- **SFX gesture jari** (dari gambar referensi user): 1 jari = klik/beep, 2 = swoosh, 3 = synth naik + ledakan (sekali saat gesture dibentuk, jeda ulang 0.3 s); 4 = beat loop selama ditahan. Pemetaan gesture → nama WAV / mode (`once`/`loop`) / volume di `finger_sfx` pada `config.yaml`; file di `assets/audio/`, fallback ke placeholder bawaan; nama yang tak ditemukan diabaikan dengan peringatan. Format MP3/WAV/OGG/FLAC via `soundfile`.
- Suara user saat ini (MP3 download, **tidak di-commit**, `assets/audio/*.mp3` di `.gitignore`): 1 jari = finger1-sound, 2 = finger2-sound, 3 = akh, 4 = guru-hentikan-guru (0.85–12 s); semua `once`. Suara `once` **berhenti saat gesture jarinya dilepas** (ke gesture lain / kepal / tangan turun / efek dimatikan), fade 1 blok. **Satu kanal SFX**: suara one-shot baru (jari maupun orb: activation/blast/power-down) memotong suara sebelumnya (fade 1 blok ±5 ms), jadi tidak pernah berbunyi bersamaan; ganti jari = suara lama berhenti, suara jari baru mulai.
- Suara jari baru diputar setelah gesture ditahan `finger_sfx.hold` 0.15 s (total ±0.27 s dengan debounce), dan tidak diputar selama orb charging/blast/cooldown 1 s setelah blast (`OrbMachine.busy`): hitungan jari yang dilewati saat tangan membuka/menutup (mis. melepas kepalan untuk blast) tidak ikut berbunyi.
- Gesture jari saat orb aktif → orb memudar **tanpa** suara power-down. Gesture jari tidak memunculkan orb. Belum ada visual untuk gesture jari (hanya label di preview).
- Checkpoint 4: mic **webcam (FHD Camera)** (WASAPI InputStream; semula USB Audio CODEC, tapi diukur: input USB hanya desis/dengung — 28% energi di pita suara — sedangkan mic webcam 94% dan juga mic default Windows) + SFX → OutputStream "CABLE Input" → mic "CABLE Output" di Meet. Mic dan VB-CABLE beda clock → FIFO dibatasi 60 ms (audio tertua dibuang, latency tidak merayap). SFX ke headphone dan ke meeting lewat dua Mixer yang menerima perintah sama (tanpa menyalin audio antar-thread). `M` membisukan SFX di keduanya, suara user tetap lewat. VB-CABLE belum terpasang → `MEET AUDIO OFF`, aplikasi tetap jalan.
- VB-CABLE diinstal manual oleh user saat Checkpoint 4. Jika Meet memotong SFX, matikan noise cancellation Meet.

## Kontrol (jendela preview)
- `Space` efek on/off · `D` skeleton debug · `M` mute SFX · `Q`/`Esc` keluar.
- Threshold gesture di-tuning dari pengukuran user: fist ≈ 0.32, open ≈ 1.10 → `fist_below 0.55`, `open_above 0.8`.

## Struktur kode
Ramping: `main.py`, `tracking.py`, `gestures.py`, `orb.py`, `audio.py`, `config.yaml`, `tests/`. Pecah file hanya saat efek kedua muncul. Repo: github.com/lfarabi/meetcam (private, GPL-3.0), branch `main`.

## Checkpoint (user menguji dengan tangan sendiri)
1. Preview + skeleton + label gesture + FPS.
2. Orb + charging + blast.
3. Virtual camera di Google Meet.
4. VB-CABLE + SFX + mic di Meet.
