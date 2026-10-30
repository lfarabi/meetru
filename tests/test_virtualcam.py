# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 lfarabi
import ctypes
import os
import sys

import cv2
import numpy as np
import pytest

import virtualcam
from virtualcam import STATE_READY, STATE_STARTING, STATE_STOPPING, Header, ObsVirtualCamera, VirtualCameraError

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the OBS virtual camera exists on Windows only")
SIZE = (64, 48)


class Reader:
    """What the camera side does: opens the queue read-only and reads the newest slot back."""

    def __init__(self, name):
        self.k = virtualcam._kernel32()
        self.handle = self.k.OpenFileMappingW(virtualcam.FILE_MAP_READ, 0, name)
        assert self.handle, "the writer's queue should exist"
        self.view = self.k.MapViewOfFile(self.handle, virtualcam.FILE_MAP_READ, 0, 0, 0)
        self.header = Header.from_address(self.view)

    def frame(self):
        h, w = self.header.cy, self.header.cx
        slot = self.header.offsets[self.header.read_idx % virtualcam.SLOTS]
        nv12 = np.ctypeslib.as_array((ctypes.c_uint8 * (w * h * 3 // 2)).from_address(
            self.view + slot + virtualcam.SLOT_HEADER)).reshape(h * 3 // 2, w).copy()
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

    def close(self):
        self.k.UnmapViewOfFile(self.view)
        self.k.CloseHandle(self.handle)


@pytest.fixture
def queue_name():
    return f"MeetruTestQueue-{os.getpid()}"  # never the real camera's queue, which OBS or a running Meetru may hold


@pytest.fixture
def camera(queue_name):
    cam = ObsVirtualCamera(*SIZE, 30, name=queue_name, check_driver=False)
    yield cam
    cam.close()


@pytest.fixture
def reader(camera, queue_name):
    r = Reader(queue_name)
    yield r
    r.close()   # a reader that is left open would keep the queue alive and block the next writer


def picture(seed=0, w=SIZE[0], h=SIZE[1]):
    """A smooth, colourful test picture (4:2:0 colour halves the colour detail, so no sharp edges)."""
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    return (np.stack([x, y, 1 - x], axis=2) * (110 + 30 * seed) + 20).astype(np.uint8)


def test_the_header_and_the_slots_are_laid_out_like_obs():
    assert ctypes.sizeof(Header) == 80
    offsets, size = virtualcam.layout(1280, 720)
    slot = 1280 * 720 * 3 // 2 + virtualcam.SLOT_HEADER
    assert offsets == [96, 96 + slot, 96 + 2 * slot] and size == 96 + 3 * slot
    assert all(o % 32 == 0 for o in offsets) and size % 32 == 0
    odd, odd_size = virtualcam.layout(66, 50)  # sizes that don't land on 32 bytes still align
    assert all(o % 32 == 0 for o in odd) and odd_size % 32 == 0 and odd[1] - odd[0] >= 66 * 50 * 3 // 2 + 32


@windows_only
def test_frames_reach_the_queue_as_nv12_and_rotate_through_three_slots(camera, reader):
    head = reader.header
    assert (head.state, head.cx, head.cy, head.interval, head.type) == (STATE_STARTING, 64, 48, 333333, 0)
    slots = []
    for i in range(1, 5):
        p = picture(seed=i)
        camera.send(p)
        assert (head.write_idx, head.read_idx, head.state) == (i, i, STATE_READY)
        assert np.abs(reader.frame().astype(int) - p.astype(int)).mean() < 3   # the slot holds this very picture
        slots.append(head.read_idx % virtualcam.SLOTS)
    assert slots == [1, 2, 0, 1]   # round-robin: the 4th frame goes back into the first slot


@windows_only
def test_the_picture_is_upright_and_the_colours_are_in_the_right_channels(camera, reader):
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[:24, :32] = (255, 0, 0)    # top-left blue (BGR)
    frame[24:, 32:] = (0, 0, 255)    # bottom-right red
    camera.send(frame)
    back = reader.frame()
    assert back[6, 8, 0] > 200 and back[6, 8, 2] < 60      # top-left is still blue, so not flipped or swapped
    assert back[40, 56, 2] > 200 and back[40, 56, 0] < 60  # bottom-right is red


@windows_only
def test_a_second_writer_is_refused_until_the_first_has_gone(queue_name):
    first = ObsVirtualCamera(*SIZE, 30, name=queue_name, check_driver=False)
    try:
        with pytest.raises(VirtualCameraError, match="already sending"):
            ObsVirtualCamera(*SIZE, 30, name=queue_name, check_driver=False)
    finally:
        first.close()
    first.close()   # closing twice is harmless
    ObsVirtualCamera(*SIZE, 30, name=queue_name, check_driver=False).close()


@windows_only
def test_closing_tells_a_reader_that_still_holds_the_queue_that_it_ended(camera, reader):
    camera.send(picture())
    camera.close()
    assert reader.header.state == STATE_STOPPING   # OBS's camera sees this and lets go of the queue
    with pytest.raises(VirtualCameraError, match="closed"):
        camera.send(picture())


@windows_only
def test_wrong_frames_and_sizes_are_refused(camera, queue_name):
    with pytest.raises(VirtualCameraError, match="even"):
        ObsVirtualCamera(65, 48, 30, name=queue_name + "-odd", check_driver=False)
    with pytest.raises(ValueError):
        camera.send(np.zeros((48, 62, 3), np.uint8))
    with pytest.raises(ValueError):
        camera.send(np.zeros((48, 64, 3), np.float32))


def test_without_obs_installed_or_off_windows_it_says_so(monkeypatch, queue_name):
    monkeypatch.setattr(virtualcam, "driver_installed", lambda: False)
    if sys.platform == "win32":
        with pytest.raises(VirtualCameraError, match="not installed"):
            ObsVirtualCamera(*SIZE, 30, name=queue_name)
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(VirtualCameraError, match="only supported on Windows"):
        ObsVirtualCamera(*SIZE, 30, name=queue_name)
    assert virtualcam.driver_installed() is False
