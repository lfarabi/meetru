"""The window's texts in Indonesian and English. `t(key)` gives the text in the current language; the language
is chosen once at start (`set_language`), not switched live."""
from PySide6.QtCore import QLocale

STRINGS = {
    "id": {
        "tab_settings": "Pengaturan",
        "btn_effects": "Efek (Spasi)",
        "btn_mute": "Bisukan suara (M)",
        "btn_room": "Foto ruangan (B)",
        "btn_skeleton": "Kerangka tangan (D)",
        "state_on": "NYALA",
        "state_off": "MATI",
        "light_vcam": "Kamera virtual: {state}",
        "light_meeting": "Suara meeting: {state}",
        "light_fps": "{fps} FPS",
        "tip_vcam_off": "Pasang OBS Studio (di dalamnya ada 'OBS Virtual Camera'), lalu jalankan ulang MeetCam",
        "tip_meeting_off": "Pasang VB-CABLE, jalankan ulang MeetCam, lalu pilih 'CABLE Output' "
                           "sebagai mikrofon di Meet/Zoom",
        "badge_invisible": "Tak terlihat",
        "set_camera": "Kamera",
        "set_mic": "Mikrofon",
        "set_speaker": "Speaker (yang kamu dengar)",
        "set_meeting_output": "Keluaran meeting (yang didengar peserta lain)",
        "set_vcam": "Kirim ke kamera virtual",
        "set_language": "Bahasa",
        "set_mirror": "Cerminkan pratinjau",
        "device_default": "(bawaan)",
        "camera_n": "Kamera {n}",
        "lang_auto": "Otomatis",
        "restart_for_language": "Jalankan ulang MeetCam supaya bahasanya berganti.",
        "camera_failed": "Kamera tidak bisa dipakai. Tutup aplikasi lain yang memakainya (misalnya NVIDIA "
                         "Broadcast), lalu pilih kamera lagi di Pengaturan.",
        "tracker_failed": "Model tangan tidak ditemukan (models/hand_landmarker.task). "
                          "Lihat README untuk cara mengunduhnya.",
        "room_countdown": "Keluar dari layar: foto ruangan kosong diambil dalam {seconds} detik",
        "room_captured": "Foto ruangan kosong tersimpan",
        "no_room_photo": "Belum ada foto ruangan kosong: tekan Foto ruangan (B) dulu",
        "record_countdown": "Siap-siap: gerakanmu direkam dalam {seconds} detik",
        "recording": "Merekam... tahan gerakanmu",
        "problems_text": "File ini tidak bisa dibaca, jadi disisihkan sebagai .broken dan pengaturan awal "
                         "dipakai: {files}",
    },
    "en": {
        "tab_settings": "Settings",
        "btn_effects": "Effects (Space)",
        "btn_mute": "Mute sounds (M)",
        "btn_room": "Room photo (B)",
        "btn_skeleton": "Skeleton (D)",
        "state_on": "ON",
        "state_off": "OFF",
        "light_vcam": "Virtual camera: {state}",
        "light_meeting": "Meeting audio: {state}",
        "light_fps": "{fps} FPS",
        "tip_vcam_off": "Install OBS Studio (it adds 'OBS Virtual Camera'), then restart MeetCam",
        "tip_meeting_off": "Install VB-CABLE, restart MeetCam, then choose 'CABLE Output' "
                           "as the microphone in Meet/Zoom",
        "badge_invisible": "Invisible",
        "set_camera": "Camera",
        "set_mic": "Microphone",
        "set_speaker": "Speaker (what you hear)",
        "set_meeting_output": "Meeting output (what the others hear)",
        "set_vcam": "Send to the virtual camera",
        "set_language": "Language",
        "set_mirror": "Mirror the preview",
        "device_default": "(default)",
        "camera_n": "Camera {n}",
        "lang_auto": "Auto",
        "restart_for_language": "Restart MeetCam to change the language.",
        "camera_failed": "The camera can't be used. Close other apps that use it (for example NVIDIA "
                         "Broadcast), then pick the camera again in Settings.",
        "tracker_failed": "The hand model is missing (models/hand_landmarker.task). "
                          "See the README for how to download it.",
        "room_countdown": "Leave the frame: photo of the empty room in {seconds}s",
        "room_captured": "Empty-room photo saved",
        "no_room_photo": "No empty-room photo yet: press Room photo (B) first",
        "record_countdown": "Get ready: your gesture is recorded in {seconds}s",
        "recording": "Recording... hold your pose",
        "problems_text": "These files could not be read, so they were set aside as .broken and defaults "
                         "are used: {files}",
    },
}

_lang = "en"


def set_language(code: str | None):
    """"id" or "en"; None follows the system language ("id" if it is Indonesian, else "en")."""
    global _lang
    if code is None:
        code = "id" if QLocale.system().name().startswith("id") else "en"
    _lang = code if code in STRINGS else "en"


def t(key: str, **fmt) -> str:
    """The text for `key` with `{fields}` filled in; an unknown key (say, from a newer engine) is shown as is."""
    text = STRINGS[_lang].get(key, key)
    return text.format(**fmt) if fmt else text
