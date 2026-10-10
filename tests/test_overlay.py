from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import overlay
from jutsu import Clip
from overlay import OverlayPlayer, RELEASE_FADE, image_clip, place, text_clip, visual_clip


def solid(w, h):
    return Clip([(np.full((h, w, 3), 200, np.uint8), np.full((h, w), 255, np.uint8))], 30.0, 0, (w / 2, h / 2), (0, 0), w)


def test_grid_positions_are_as_seen_in_the_preview():
    c = solid(100, 100)                                     # size 0.25 of a 400-high frame → scale 1.0
    assert place(c, "top_left", 0.25, 800, 400, None, preview=True)[:2] == (16, 16)
    assert place(c, "top_left", 0.25, 800, 400, None, preview=False)[:2] == (800 - 16 - 100, 16)
    assert place(c, "bottom_right", 0.25, 800, 400, None, preview=True)[:2] == (800 - 16 - 100, 400 - 16 - 100)
    assert place(c, "center", 0.25, 800, 400, None, preview=True)[:2] == (350, 150)
    assert place(c, "no_such_spot", 0.25, 800, 400, None, preview=True)[:2] == (350, 150)   # hand-edited file


def test_hand_position_follows_the_palm_mirrored_for_the_preview():
    x0, y0, s = place(solid(100, 100), "hand", 2.0, 800, 400, (200, 300, 50), preview=False)
    assert (x0, y0, s) == (150, 250, 1.0)
    assert place(solid(100, 100), "hand", 2.0, 800, 400, (200, 300, 50), preview=True)[:2] == (550, 250)
    assert place(solid(100, 100), "hand", 2.0, 800, 400, None, preview=True) is None


def test_full_keeps_the_clip_framing():
    c = Clip(solid(50, 40).frames, 30.0, 0, (25, 20), (10, 20), 200)   # cropped from a 200-wide source
    assert place(c, "full", 1.0, 800, 450, None, preview=False) == (40, 80, 4.0)
    assert place(c, "full", 1.0, 800, 450, None, preview=True) == (800 - 40 - 200, 80, 4.0)


def test_gif_and_unicode_named_png_load_with_alpha(tmp_path):
    frames = [Image.new("RGB", (8, 6), c) for c in ("red", "green", "blue")]
    frames[0].save(tmp_path / "a.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    gif = image_clip(tmp_path / "a.gif")
    assert len(gif.frames) == 3 and gif.fps == pytest.approx(10)
    assert gif.origin == (0, 0) and gif.source_width == 8 and gif.peak == 0 and gif.anchor == (4, 3)

    png = np.zeros((6, 8, 4), np.uint8)
    png[:, :4] = (255, 0, 0, 255)                           # opaque red left half, transparent right half
    Image.fromarray(png, "RGBA").save(tmp_path / "é.png")
    bgr, alpha = image_clip(tmp_path / "é.png").frames[0]
    assert alpha.min() == 0 and alpha.max() == 255
    assert tuple(bgr[0, 0]) == (0, 0, 255)                  # red, in BGR order


def test_phone_photos_are_upright_and_single_frame(tmp_path):
    img = Image.new("RGB", (40, 20), "red")
    img.paste("blue", (20, 0, 40, 20))                      # red left half, blue right half
    exif = Image.Exif()
    exif[0x0112] = 6                                        # Orientation: show rotated 90° clockwise
    img.save(tmp_path / "rot.jpg", exif=exif)
    bgr, _ = image_clip(tmp_path / "rot.jpg").frames[0]
    assert bgr.shape[:2] == (40, 20)                        # upright: taller than wide, red on top
    assert bgr[2, 10, 2] > 200 and bgr[2, 10, 0] < 50 and bgr[-3, 10, 0] > 200 and bgr[-3, 10, 2] < 50

    img.save(tmp_path / "phone.jpg", format="MPO", save_all=True, append_images=[img.resize((10, 5))])
    with Image.open(tmp_path / "phone.jpg") as im:
        assert im.format == "MPO" and im.n_frames == 2      # a main picture plus a smaller second one
    assert len(image_clip(tmp_path / "phone.jpg").frames) == 1


def test_big_images_are_shrunk_but_small_ones_are_not_enlarged(tmp_path):
    Image.new("RGB", (2000, 1500), "green").save(tmp_path / "big.png")
    big = image_clip(tmp_path / "big.png")
    assert big.frames[0][0].shape[:2] == (720, 960) and big.source_width == 960
    Image.new("RGB", (300, 200), "green").save(tmp_path / "small.png")
    assert image_clip(tmp_path / "small.png").frames[0][0].shape[:2] == (200, 300)


def test_emoji_keep_their_own_colours(qapp):
    from PySide6.QtGui import QFontDatabase
    font = Path("C:/Windows/Fonts/seguiemj.ttf")            # offscreen has no fonts of its own
    if not font.is_file() or QFontDatabase.addApplicationFont(str(font)) < 0:
        pytest.skip("no colour emoji font available")
    bgr, alpha = text_clip("😀", "#ff00ff").frames[0]   # magenta fill, dark outline
    assert bgr[alpha == 255][:, 1].max() > 100              # something with green in it: not a one-colour silhouette
    bgr, alpha = text_clip("WOW!", "#ff00ff").frames[0]
    assert bgr[alpha == 255][:, 1].max() <= 20              # plain text is only fill and outline: no green


def test_text_renders_a_visible_sprite(qapp):
    c = text_clip("WOW!", "#ffd400")
    bgr, alpha = c.frames[0]
    assert alpha.max() == 255 and alpha.min() == 0
    assert (bgr[alpha == 255] == (0, 212, 255)).all(axis=1).any()   # the #ffd400 fill, in BGR order
    assert c.source_width == bgr.shape[1] and c.origin == (0, 0)
    with pytest.raises(ValueError):
        text_clip("x", "not a colour")


def test_visual_clip_dispatches_and_raises_on_failure(tmp_path, qapp, monkeypatch):
    Image.new("RGBA", (4, 4)).save(tmp_path / "s.png")
    assert len(visual_clip({"type": "image"}, tmp_path / "s.png").frames) == 1
    assert visual_clip({"type": "text", "text": "hi"}, None).frames[0][1].max() == 255
    monkeypatch.setattr(overlay.jutsu, "load_clip", lambda path: ("video", path))
    assert visual_clip({"type": "video"}, tmp_path / "s.png") == ("video", tmp_path / "s.png")
    for kind in ("image", "video"):
        for path in (None, tmp_path / "gone.png"):
            with pytest.raises(FileNotFoundError):
                visual_clip({"type": kind}, path)
    with pytest.raises(ValueError):
        visual_clip({"type": "orb"}, None)


def test_player_held_fades_on_release_and_timed_ends_by_itself():
    p = OverlayPlayer(solid(10, 10), {"duration": "held"}); p.start(0.0); p.update(5.0); assert p.busy
    p.release(5.0); p.update(5.0 + RELEASE_FADE + 0.01); assert not p.busy
    q = OverlayPlayer(solid(10, 10), {"duration": 2.0}); q.start(0.0); q.update(1.9); assert q.busy; q.update(2.01); assert not q.busy
    r = OverlayPlayer(solid(10, 10), {"duration": "held"}); r.start(0.0, seconds=2.0); r.update(2.01); assert not r.busy
    s = OverlayPlayer(solid(10, 10), {"duration": 2.0}); s.start(0.0); s.release(0.5); s.update(1.0); assert s.busy   # timed ones ignore release


def test_player_draws_looping_frames_and_fades_after_release():
    dark, light = (np.full((10, 10, 3), v, np.uint8) for v in (0, 200))
    opaque = np.full((10, 10), 255, np.uint8)
    p = OverlayPlayer(Clip([(dark, opaque), (light, opaque)], 10.0, 0, (5, 5), (0, 0), 10),
                      {"position": "center", "size": 0.25, "duration": "held"})   # 100 px sprite on a 400-high frame
    frame = np.full((400, 800, 3), 50, np.uint8)
    p.draw(frame, 0.0, None, preview=True); assert (frame == 50).all()           # idle: untouched
    p.start(1.0)
    for t, want in ((1.05, 0), (1.15, 200), (1.25, 0)):                            # frame index loops
        frame[:] = 50; p.draw(frame, t, None, preview=True); assert frame[200, 400, 0] == want
    p.release(1.05); frame[:] = 50; p.draw(frame, 1.05 + RELEASE_FADE / 2, None, preview=True)
    assert frame[200, 400, 0] == pytest.approx(125, abs=1)                        # half way from 200 to 50
