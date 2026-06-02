"""In-memory attendance registry."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from services.logger import get_logger


class AttendanceCollector:
    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        self._lock = threading.RLock()
        self._registry: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _timestamp(value: Any = None) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str) and value:
            return value
        return datetime.now().isoformat()

    def record_detection(self, student_id: str, confidence: float, timestamp: Any = None) -> None:
        if not student_id:
            return

        now = self._timestamp(timestamp)
        with self._lock:
            existing = self._registry.get(student_id)
            if existing is None:
                self._registry[student_id] = {
                    "first_seen": now,
                    "last_seen": now,
                    "highest_confidence": float(confidence),
                }
                self.logger.info("Recorded first attendance detection for %s.", student_id)
                return

            existing["last_seen"] = now
            existing["highest_confidence"] = max(float(confidence), float(existing.get("highest_confidence", 0.0)))
            self.logger.info("Updated attendance detection for %s.", student_id)

    def generate_batch(self, session_id: str, device_id: str) -> dict[str, Any]:
        with self._lock:
            records = [
                {
                    "matric_no": student_id,
                    "confidence": round(float(data["highest_confidence"]), 4),
                    "first_seen": data["first_seen"],
                    "last_seen": data["last_seen"],
                }
                for student_id, data in sorted(self._registry.items(), key=lambda item: item[1]["first_seen"])
            ]

        return {
            "session_id": session_id,
            "device_id": device_id,
            "records": records,
        }

    def clear_batch(self) -> None:
        with self._lock:
            self._registry.clear()
        self.logger.info("Attendance registry cleared.")

    def get_record_count(self) -> int:
        with self._lock:
            return len(self._registry)
