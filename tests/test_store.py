from store import DEFAULT_JURUS, DEFAULT_SETTINGS, Store


def test_missing_files_give_defaults(tmp_path):
    s = Store(tmp_path)
    assert s.load_settings() == DEFAULT_SETTINGS and s.load_jurus() == DEFAULT_JURUS and s.load_gestures() == []


def test_loaded_defaults_are_copies(tmp_path):
    s = Store(tmp_path)
    s.load_jurus()[0]["name"] = "changed"
    s.load_settings()["language"] = "xx"
    assert s.load_jurus() == DEFAULT_JURUS and s.load_settings() == DEFAULT_SETTINGS
    assert DEFAULT_JURUS[0]["name"] == "Bola Energi" and DEFAULT_SETTINGS["language"] is None


def test_round_trip(tmp_path):
    s = Store(tmp_path)
    s.save_settings({**DEFAULT_SETTINGS, "language": "en"}); s.save_jurus(DEFAULT_JURUS[:1]); s.save_gestures([{"name": "ok"}])
    s2 = Store(tmp_path)
    assert s2.load_settings()["language"] == "en" and s2.load_jurus() == DEFAULT_JURUS[:1] and s2.load_gestures() == [{"name": "ok"}]


def test_broken_file_is_set_aside_and_defaults_used(tmp_path):
    (tmp_path / "jurus.yaml").write_text("- [unclosed", encoding="utf-8")
    s = Store(tmp_path)
    assert s.load_jurus() == DEFAULT_JURUS
    assert (tmp_path / "jurus.yaml.broken").exists() and s.problems


def test_import_media_copies_sanitises_and_dedupes(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    a = src / "stiker é.png"; a.write_bytes(b"A")
    s = Store(tmp_path / "data")
    ref = s.import_media(a)
    assert ref == "media/stiker_.png" and s.resolve(ref).read_bytes() == b"A"
    assert s.import_media(a) == ref                      # same content → same file
    b = src / "stiker é.png"; b.write_bytes(b"B")
    assert s.import_media(b) == "media/stiker__2.png"   # same name, different content
    assert s.resolve("builtin:finger_1") is None
    assert s.resolve("") is None
