"""Local attendance export generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import ATTENDANCE_EXPORT_DIR
from services.logger import get_logger, log_event


class ExportService:
    def __init__(self, export_dir: Path = ATTENDANCE_EXPORT_DIR) -> None:
        self.export_dir = export_dir
        self.logger = get_logger(__name__)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_code(course_code: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in course_code or "UNKNOWN")

    @staticmethod
    def _safe_session_id(session_id: str | None) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(session_id or "UNKNOWN"))

    def _base_filename(self, session_id: str, course_code: str, timestamp: datetime | None = None) -> str:
        stamp = timestamp or datetime.now()
        return f"{self._safe_session_id(session_id)}_{self._safe_code(course_code)}_{stamp.strftime('%Y-%m-%d_%H-%M')}"

    def _unique_path(self, base_name: str, extension: str) -> Path:
        candidate = self.export_dir / f"{base_name}.{extension}"
        counter = 1
        while candidate.exists():
            candidate = self.export_dir / f"{base_name}_{counter}.{extension}"
            counter += 1
        return candidate

    def export_batch(self, payload: dict[str, Any], course_code: str, session_id: str | None = None) -> list[Path]:
        records = payload.get("records", [])
        if not records:
            self.logger.info("Skipping export because the batch is empty.")
            return []

        base_name = self._base_filename(session_id or str(payload.get("session_id") or "UNKNOWN"), course_code)
        frame = pd.DataFrame(
            [
                {
                    "student_id": record.get("matric_no") or record.get("student_id"),
                    "session_id": payload.get("session_id"),
                    "first_seen": record.get("first_seen"),
                    "last_seen": record.get("last_seen"),
                    "highest_confidence": record.get("confidence"),
                    "device_id": payload.get("device_id"),
                }
                for record in records
            ]
        )

        written_paths: list[Path] = []

        csv_path = self._unique_path(base_name, "csv")
        frame.to_csv(csv_path, index=False)
        written_paths.append(csv_path)

        xlsx_path = self._unique_path(base_name, "xlsx")
        frame.to_excel(xlsx_path, index=False)
        written_paths.append(xlsx_path)

        log_event(
            self.logger,
            "ATTENDANCE_EXPORTED",
            csv=csv_path.name,
            xlsx=xlsx_path.name,
            records=len(records),
            session_id=payload.get("session_id") or session_id or "UNKNOWN",
        )
        return written_paths
