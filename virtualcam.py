# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 lfarabi
"""Meetru's own writer for the "OBS Virtual Camera" (Windows); it replaces the pyvirtualcam library.

OBS Studio installs a DirectShow camera, "OBS Virtual Camera", that shows whatever a program puts into a shared-memory
queue named OBSVirtualCamVideo: a small header, then three NV12 frame slots used round-robin. This module writes that
queue. The layout follows OBS Studio's plugins/win-dshow/shared-memory-queue.{h,c} (GPL-2.0-or-later, the OBS Project,
https://github.com/obsproject/obs-studio); the code here is written from scratch and contains nothing from pyvirtualcam
(GPL-2.0 without "or later", which cannot be combined with GPL-3.0 or Apache-2.0 code in one distribution).

Only Windows with OBS Studio's camera installed is supported so far; anywhere else opening raises VirtualCameraError
and the app runs with the preview only.
"""
import ctypes
import logging
import sys
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

QUEUE_NAME = "OBSVirtualCamVideo"
DEVICE_NAME = "OBS Virtual Camera"
DRIVER_CLSID = "{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"  # the DirectShow filter OBS registers as "OBS Virtual Camera"
SLOTS, SLOT_HEADER, ALIGN = 3, 32, 32  # three frame slots, each with a 32-byte header whose first 8 bytes are a timestamp
STATE_STARTING, STATE_READY, STATE_STOPPING = 1, 2, 3
PAGE_READWRITE, FILE_MAP_READ, FILE_MAP_ALL_ACCESS = 0x04, 0x04, 0xF001F


class VirtualCameraError(RuntimeError):
    """The virtual camera can't be opened: not installed, already in use, or not supported on this system."""


class Header(ctypes.Structure):
    """The start of the shared memory, as OBS defines it (struct queue_header); `interval` is in 100 ns units."""
    _fields_ = [("write_idx", ctypes.c_uint32), ("read_idx", ctypes.c_uint32), ("state", ctypes.c_uint32),
                ("offsets", ctypes.c_uint32 * SLOTS), ("type", ctypes.c_uint32),
                ("cx", ctypes.c_uint32), ("cy", ctypes.c_uint32), ("interval", ctypes.c_uint64),
                ("reserved", ctypes.c_uint32 * 8)]


def layout(cx: int, cy: int) -> tuple[list[int], int]:
    """Where the three frame slots start in the shared memory, and its total size (everything 32-byte aligned)."""
    frame = cx * cy * 3 // 2  # NV12: a full-size Y plane, then interleaved U and V at half size
    align = lambda n: (n + ALIGN - 1) & ~(ALIGN - 1)  # noqa: E731
    size, offsets = align(ctypes.sizeof(Header)), []
    for _ in range(SLOTS):
        offsets.append(size)
        size = align(size + SLOT_HEADER + frame)
    return offsets, size


def driver_installed() -> bool:
    """Is OBS Studio's virtual camera registered with Windows? (The 64-bit view: where 64-bit apps like Chrome look.)"""
    if sys.platform != "win32":
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "CLSID\\" + DRIVER_CLSID, 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
            return True
    except OSError:
        return False


_k32 = None


def _kernel32():
    global _k32
    if _k32 is None:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        k.OpenFileMappingW.restype = ctypes.c_void_p
        k.CreateFileMappingW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                         ctypes.c_uint32, ctypes.c_wchar_p]
        k.CreateFileMappingW.restype = ctypes.c_void_p
        k.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t]
        k.MapViewOfFile.restype = ctypes.c_void_p
        k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        k.UnmapViewOfFile.restype = ctypes.c_int
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        k.CloseHandle.restype = ctypes.c_int
        _k32 = k
    return _k32


class ObsVirtualCamera:
    """Sends BGR frames to the OBS Virtual Camera. The size must be even (NV12 halves the colour planes).

    `name` and `check_driver` exist for the tests, which use a private queue instead of the real camera's.
    """
    device = DEVICE_NAME

    def __init__(self, width: int, height: int, fps: int, name: str = QUEUE_NAME, check_driver: bool = True):
        self.width, self.height, self.fps = width, height, fps
        self._view = self._handle = self._mem = self._header = None
        if sys.platform != "win32":
            raise VirtualCameraError("the virtual camera is only supported on Windows so far")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise VirtualCameraError(f"the picture size {width}x{height} must be even")
        if check_driver and not driver_installed():
            raise VirtualCameraError("OBS Studio's virtual camera is not installed (install OBS Studio once)")
        k = _kernel32()
        offsets, size = layout(width, height)
        busy = k.OpenFileMappingW(FILE_MAP_READ, 0, name)  # a writer exists (or a reader still holds the old queue)
        if busy:
            k.CloseHandle(busy)
            raise VirtualCameraError("another program is already sending to the virtual camera "
                                     "(OBS's own Start Virtual Camera? stop it or close OBS)")
        handle = k.CreateFileMappingW(ctypes.c_void_p(-1), None, PAGE_READWRITE, 0, size, name)
        if not handle:
            raise VirtualCameraError(f"could not create the shared memory: {ctypes.WinError(ctypes.get_last_error())}")
        view = k.MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, 0)
        if not view:
            error = ctypes.WinError(ctypes.get_last_error())
            k.CloseHandle(handle)
            raise VirtualCameraError(f"could not map the shared memory: {error}")
        self._k, self._handle, self._view, self._offsets = k, handle, view, offsets
        self._mem = (ctypes.c_uint8 * size).from_address(view)
        head = self._header = Header.from_address(view)  # the memory starts zeroed
        head.cx, head.cy, head.interval = width, height, 10_000_000 // max(fps, 1)
        for i, offset in enumerate(offsets):
            head.offsets[i] = offset
        head.state = STATE_STARTING  # becomes READY with the first frame

    def send(self, frame: np.ndarray):
        """Puts one BGR frame (height x width x 3, uint8) where the camera picks it up."""
        if self._view is None:
            raise VirtualCameraError("the virtual camera is closed")
        w, h = self.width, self.height
        if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected a {w}x{h} BGR uint8 frame, got {frame.shape} {frame.dtype}")
        i420 = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)  # rows: Y plane, then the U plane, then the V plane
        head = self._header
        number = (head.write_idx + 1) & 0xFFFFFFFF
        head.write_idx = number
        slot = self._offsets[number % SLOTS]
        ctypes.c_uint64.from_address(self._view + slot).value = time.monotonic_ns()
        y = np.ndarray((h, w), np.uint8, self._mem, slot + SLOT_HEADER)
        uv = np.ndarray((h // 2, w), np.uint8, self._mem, slot + SLOT_HEADER + w * h)
        y[:] = i420[:h]
        uv[:, 0::2] = i420[h:h + h // 4].reshape(h // 2, w // 2)  # NV12 interleaves U and V
        uv[:, 1::2] = i420[h + h // 4:].reshape(h // 2, w // 2)
        head.read_idx = number  # the camera reads the slot this points at
        head.state = STATE_READY

    def close(self):
        if self._view is None:
            return
        self._header.state = STATE_STOPPING  # readers that still hold the queue see it end
        self._mem = self._header = None
        self._k.UnmapViewOfFile(self._view)
        self._k.CloseHandle(self._handle)
        self._view = self._handle = None
