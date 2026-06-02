"""Thin wrapper around OpenCV camera capture."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2

from config import CAMERA_INDEX
from services.logger import get_logger


class CameraStream:
    def __init__(self, camera_index: int = CAMERA_INDEX) -> None:
        self.camera_index = camera_index
        self.logger = get_logger(__name__)
        self.capture: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if self.capture is not None and self.capture.isOpened():
            return True

        self.capture = cv2.VideoCapture(self.camera_index)
        if self.capture is None or not self.capture.isOpened():
            self.logger.warning("Unable to open camera index %s.", self.camera_index)
            return False

        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return True

    def read(self) -> tuple[bool, object | None]:
        if self.capture is None and not self.open():
            return False, None

        assert self.capture is not None
        success, frame = self.capture.read()
        if not success:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

