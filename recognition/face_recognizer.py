"""InsightFace adapter that emits student recognition callbacks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from config import (
    CONFIDENCE_THRESHOLD,
    FACE_DET_SIZE,
    INSIGHTFACE_MODEL_ROOT,
    RECOGNITION_FRAME_DELAY_SECONDS,
)
from recognition.camera_stream import CameraStream
from recognition.tracker import CentroidTracker
from services.logger import get_logger


try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - handled at runtime
    FaceAnalysis = None


@dataclass
class RecognitionHandle:
    stop_event: threading.Event
    thread: threading.Thread
    camera: CameraStream

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.camera.release()


class InsightFaceRecognizer:
    def __init__(self, embedding_client=None, event_logger=None) -> None:
        self.logger = get_logger(__name__)
        self.camera = CameraStream()
        self.tracker = CentroidTracker()
        self.embedding_client = embedding_client
        self.event_logger = event_logger
        self.face_app = self._create_face_app()

    def _create_face_app(self):
        if FaceAnalysis is None:
            self.logger.error("insightface is not available; recognition cannot start.")
            return None

        try:
            app = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_MODEL_ROOT)
            app.prepare(ctx_id=-1, det_size=FACE_DET_SIZE)
            return app
        except Exception:
            self.logger.exception("Failed to initialize InsightFace FaceAnalysis.")
            return None

    def _detect_faces(self, frame) -> list[dict[str, object]]:
        if self.face_app is None:
            return []

        faces = self.face_app.get(frame)
        detections: list[dict[str, object]] = []
        for face in faces:
            bbox = getattr(face, "bbox", None)
            embedding = getattr(face, "normed_embedding", None)
            det_score = float(getattr(face, "det_score", 0.0) or 0.0)

            if bbox is None or embedding is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "embedding": np.asarray(embedding, dtype=np.float32),
                    "confidence": det_score,
                }
            )
        return detections

    def _process_frame(self, frame, callback: Callable[[str, float], None]) -> None:
        now = datetime.now()
        detections = self._detect_faces(frame)
        if detections and self.event_logger is not None:
            try:
                self.event_logger(
                    "FACE_DETECTED",
                    "Face detected in camera frame.",
                    {"count": len(detections)},
                )
            except Exception:
                self.logger.debug("Device log sink failed during face detection.", exc_info=True)
        tracks = self.tracker.update(detections, now=now)

        for track in tracks:
            matching_detection = None
            for detection in detections:
                if tuple(detection["bbox"]) == tuple(track["bbox"]):
                    matching_detection = detection
                    break

            if matching_detection is None:
                continue

            if not self.tracker.should_recognize(
                int(track["track_id"]),
                detection_embedding=matching_detection["embedding"],
                detection_confidence=float(matching_detection.get("confidence") or 0.0),
            ):
                continue

            embedding_index = None
            if self.embedding_client is not None:
                embedding_index = self.embedding_client.get_embedding_index()

            if embedding_index is None:
                continue

            student_id, match_score = embedding_index.match(
                query_embedding=matching_detection["embedding"],
                threshold=CONFIDENCE_THRESHOLD,
            )
            if student_id is None:
                continue

            self.tracker.mark_recognized(
                int(track["track_id"]),
                student_id,
                match_score,
                embedding=matching_detection["embedding"],
                now=now,
            )
            callback(student_id, float(match_score))

    def run(self, callback: Callable[[str, float], None], stop_event: threading.Event) -> None:
        self.logger.info("Recognition loop started.")
        while not stop_event.is_set():
            if self.face_app is None:
                time.sleep(2)
                self.face_app = self._create_face_app()
                continue

            if not self.camera.open():
                time.sleep(2)
                continue

            success, frame = self.camera.read()
            if not success or frame is None:
                self.logger.debug("Camera frame unavailable; retrying.")
                time.sleep(0.1)
                continue

            try:
                self._process_frame(frame, callback)
            except Exception:
                self.logger.exception("Unhandled error while processing a camera frame.")

            time.sleep(RECOGNITION_FRAME_DELAY_SECONDS)

        self.camera.release()
        self.logger.info("Recognition loop stopped.")


def start_recognition(
    callback: Callable[[str, float], None],
    stop_event: Optional[threading.Event] = None,
    embedding_client=None,
    event_logger=None,
) -> RecognitionHandle:
    """Start the recognition pipeline on a daemon thread.

    The callback receives (matric_no, confidence) whenever the recognizer
    confidently matches a student identity.
    """

    logger = get_logger(__name__)
    stop_event = stop_event or threading.Event()
    recognizer = InsightFaceRecognizer(embedding_client=embedding_client, event_logger=event_logger)

    thread = threading.Thread(
        target=recognizer.run,
        args=(callback, stop_event),
        name="face-recognition",
        daemon=True,
    )
    thread.start()
    logger.info("Recognition thread started.")
    return RecognitionHandle(stop_event=stop_event, thread=thread, camera=recognizer.camera)
