"""Lightweight centroid tracker for face tracks."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np

from config import CONFIDENCE_THRESHOLD, TRACK_EXPIRY_SECONDS
from services.logger import get_logger, log_event


TRACK_MISMATCH_THRESHOLD = 0.55


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, expiry_seconds: int = TRACK_EXPIRY_SECONDS) -> None:
        self.logger = get_logger(__name__)
        self.max_distance = max_distance
        self.expiry_seconds = expiry_seconds
        self._next_track_id = 1
        self._tracks: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _normalize_embedding(embedding: Any) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        try:
            vector = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if vector.size == 0:
            return None
        return vector

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))

    def _create_track(self, bbox: tuple[int, int, int, int], now: datetime, embedding: Any = None) -> dict[str, Any]:
        track = {
            "track_id": self._next_track_id,
            "student_id": None,
            "confidence": 0.0,
            "last_seen": now,
            "recognition_locked": False,
            "bbox": bbox,
            "centroid": self._centroid(bbox),
            "embedding": self._normalize_embedding(embedding),
            "created_at": now,
        }
        self._tracks[self._next_track_id] = track
        log_event(self.logger, "TRACK_CREATED", track_id=self._next_track_id)
        self._next_track_id += 1
        return track

    def _unlock_track(self, track: dict[str, Any], reason: str) -> None:
        if track.get("recognition_locked"):
            track["recognition_locked"] = False
            track["student_id"] = None
            track["confidence"] = 0.0
            log_event(self.logger, "TRACK_UNLOCKED", track_id=track["track_id"], reason=reason)

    def update(self, detections: list[dict[str, Any]], now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now()
        seen_track_ids: set[int] = set()

        for detection in detections:
            bbox = detection.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            bbox_tuple = tuple(int(v) for v in bbox)
            centroid = self._centroid(bbox_tuple)
            embedding = self._normalize_embedding(detection.get("embedding"))
            detection_confidence = float(detection.get("confidence") or 0.0)

            best_track: Optional[dict[str, Any]] = None
            best_distance = self.max_distance + 1.0
            for track in self._tracks.values():
                if track["track_id"] in seen_track_ids:
                    continue
                distance = self._distance(track["centroid"], centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_track = track

            if best_track is None or best_distance > self.max_distance:
                best_track = self._create_track(bbox_tuple, now, embedding=embedding)
            else:
                best_track["bbox"] = bbox_tuple
                best_track["centroid"] = centroid
                best_track["last_seen"] = now
                if embedding is not None:
                    best_track["embedding"] = embedding

            seen_track_ids.add(int(best_track["track_id"]))

            if best_track.get("recognition_locked"):
                if detection_confidence < CONFIDENCE_THRESHOLD:
                    self._unlock_track(best_track, "low_detection_confidence")
                else:
                    locked_embedding = self._normalize_embedding(best_track.get("embedding"))
                    if locked_embedding is not None and embedding is not None:
                        similarity = self._cosine_similarity(locked_embedding, embedding)
                        if similarity < TRACK_MISMATCH_THRESHOLD:
                            self._unlock_track(best_track, "embedding_mismatch")

        expired_track_ids: list[int] = []
        for track_id, track in self._tracks.items():
            if (now - track["last_seen"]).total_seconds() > self.expiry_seconds:
                expired_track_ids.append(track_id)

        for track_id in expired_track_ids:
            log_event(self.logger, "TRACK_EXPIRED", track_id=track_id)
            self._tracks.pop(track_id, None)

        return list(self._tracks.values())

    def mark_recognized(
        self,
        track_id: int,
        student_id: str,
        confidence: float,
        embedding: Any = None,
        now: datetime | None = None,
    ) -> None:
        track = self._tracks.get(track_id)
        if track is None:
            return

        now = now or datetime.now()
        normalized_embedding = self._normalize_embedding(embedding)
        track["student_id"] = student_id
        track["confidence"] = float(confidence)
        track["last_seen"] = now
        track["recognition_locked"] = True
        if normalized_embedding is not None:
            track["embedding"] = normalized_embedding

        log_event(
            self.logger,
            "TRACK_LOCKED",
            track_id=track_id,
            matric_no=student_id,
            confidence=round(float(confidence), 4),
        )

    def should_recognize(
        self,
        track_id: int,
        detection_embedding: Any = None,
        detection_confidence: float | None = None,
    ) -> bool:
        track = self._tracks.get(track_id)
        if track is None:
            return False

        if not track.get("recognition_locked"):
            return True

        if detection_confidence is not None and float(detection_confidence) < CONFIDENCE_THRESHOLD:
            self._unlock_track(track, "low_detection_confidence")
            return True

        current_embedding = self._normalize_embedding(detection_embedding)
        locked_embedding = self._normalize_embedding(track.get("embedding"))
        if current_embedding is not None and locked_embedding is not None:
            similarity = self._cosine_similarity(current_embedding, locked_embedding)
            if similarity < TRACK_MISMATCH_THRESHOLD:
                self._unlock_track(track, "embedding_mismatch")
                return True

        log_event(self.logger, "RECOGNITION_SKIPPED", track_id=track_id, reason="track_reused")
        return False

    def get_tracks(self) -> list[dict[str, Any]]:
        return list(self._tracks.values())
