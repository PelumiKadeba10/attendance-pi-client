"""Main entrypoint for the Raspberry Pi attendance client."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from config import (
    ATTENDANCE_EXPORT_DIR,
    DEVICE_ID,
    STORAGE_DIR,
)
from recognition.face_recognizer import start_recognition
from services.api_client import BackendApiClient
from services.attendance_collector import AttendanceCollector
from services.embedding_client import EmbeddingClient
from services.export_service import ExportService
from services.logger import get_logger, initialize_logging
from services.session_manager import SessionManager
from services.upload_service import UploadService


@dataclass
class RuntimeContext:
    api_client: BackendApiClient
    session_manager: SessionManager
    embedding_client: EmbeddingClient
    collector: AttendanceCollector
    export_service: ExportService
    upload_service: UploadService
    stop_event: threading.Event


class PiAttendanceApp:
    def __init__(self) -> None:
        initialize_logging()
        self.logger = get_logger(__name__)
        self.stop_event = threading.Event()
        self._session_state_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._recognition_lock = threading.Lock()
        self._last_session_id: Optional[str] = None
        self._last_course_code: Optional[str] = None
        self._last_state: str = "IDLE"
        self._recognition_handle = None

        api_client = BackendApiClient()
        session_manager = SessionManager(api_client=api_client)
        embedding_client = EmbeddingClient(api_client=api_client)
        collector = AttendanceCollector()
        export_service = ExportService(export_dir=ATTENDANCE_EXPORT_DIR)
        upload_service = UploadService(api_client=api_client)

        self.ctx = RuntimeContext(
            api_client=api_client,
            session_manager=session_manager,
            embedding_client=embedding_client,
            collector=collector,
            export_service=export_service,
            upload_service=upload_service,
            stop_event=self.stop_event,
        )

    def _current_primary_session(self) -> Optional[dict[str, Any]]:
        return self.ctx.session_manager.get_primary_active_session()

    def _remember_session(self, session: Optional[dict[str, Any]]) -> None:
        with self._session_state_lock:
            if session:
                self._last_session_id = str(session.get("session_id")) if session.get("session_id") is not None else None
                self._last_course_code = str(session.get("course_code") or "UNKNOWN")

    def _stop_recognition(self) -> None:
        with self._recognition_lock:
            if self._recognition_handle is None:
                return
            self._recognition_handle.stop()
            self._recognition_handle = None
            self.logger.info("Recognition pipeline stopped.")

    def _ensure_recognition(self) -> None:
        with self._recognition_lock:
            if self._recognition_handle is not None:
                return
            self._recognition_handle = start_recognition(
                callback=self._recognition_callback,
                stop_event=self.stop_event,
                embedding_client=self.ctx.embedding_client,
            )
            self.logger.info("Recognition pipeline started.")

    def _flush_collector(self, session_id: str, course_code: str, reason: str) -> bool:
        with self._flush_lock:
            if self.ctx.collector.get_record_count() == 0:
                self.logger.info("No attendance records to flush (%s).", reason)
                return True

            batch = self.ctx.collector.generate_batch(session_id=session_id, device_id=DEVICE_ID)
            if not batch["records"]:
                self.logger.info("Generated empty batch during %s.", reason)
                return True

            self.logger.info(
                "Flushing %s attendance records for session %s because %s.",
                len(batch["records"]),
                session_id,
                reason,
            )

            self.ctx.export_service.export_batch(
                payload=batch,
                course_code=course_code,
                session_id=session_id,
            )

            upload_success = self.ctx.upload_service.upload_cycle(batch)
            if upload_success:
                self.ctx.collector.clear_batch()
                self.logger.info("Attendance collector cleared after successful flush.")
                return True

            self.logger.error("Flush failed; collector retained for retry.")
            return False

    def _recognition_callback(self, student_id: str, confidence: float) -> None:
        session = self._current_primary_session()
        if not session:
            self.logger.debug(
                "Dropping recognition for %s because no active session is available.",
                student_id,
            )
            return

        self.ctx.collector.record_detection(student_id=student_id, confidence=confidence)

    def _session_context(self) -> tuple[Optional[dict[str, Any]], str, Optional[str], Optional[str]]:
        session = self.ctx.session_manager.get_primary_active_session()
        state = self.ctx.session_manager.get_state()
        session_id = None
        course_code = None

        if session is not None:
            session_id = str(session.get("session_id")) if session.get("session_id") is not None else None
            course_code = str(session.get("course_code") or "UNKNOWN")
            self._remember_session(session)
        else:
            with self._session_state_lock:
                session_id = self._last_session_id
                course_code = self._last_course_code

        return session, state, session_id, course_code

    def _run_state_machine(self) -> None:
        session, state, session_id, course_code = self._session_context()
        previous_state = self._last_state

        if state != previous_state:
            self._last_state = state

        if state == "RECORDING":
            self._ensure_recognition()
            return

        self._stop_recognition()

        if previous_state != "STOPPED" and state == "STOPPED" and session_id and course_code:
            self._flush_collector(
                session_id=session_id,
                course_code=course_code,
                reason="session stopped",
            )
            return

        if state == "READY":
            self.logger.debug("Session is ready but not yet recording.")
            return

        if state == "IDLE":
            self.logger.debug("Session manager is idle.")
            return

    def run(self) -> None:
        self.logger.info("Starting attendance Pi client for device %s.", DEVICE_ID)
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        self.ctx.session_manager.restore_state()
        self.ctx.session_manager.refresh_sessions()
        self.ctx.embedding_client.refresh_embeddings()
        current_session = self._current_primary_session()
        self._remember_session(current_session)
        self._last_state = self.ctx.session_manager.get_state()

        self.ctx.session_manager.start()
        self.ctx.embedding_client.start()
        self.ctx.upload_service.reload_cache()

        def _handle_signal(signum: int, _frame: Any) -> None:
            self.logger.info("Received signal %s; shutting down.", signum)
            self.stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        try:
            while not self.stop_event.is_set():
                try:
                    self._run_state_machine()
                except Exception:
                    self.logger.exception("Unhandled error in lifecycle loop.")
                time.sleep(1)
        finally:
            self.stop_event.set()
            self.ctx.session_manager.stop()
            self.ctx.embedding_client.stop()
            self._stop_recognition()
            graceful_session = self._last_session_id or (self._current_primary_session() or {}).get("session_id")
            graceful_course = self._last_course_code or (self._current_primary_session() or {}).get("course_code") or "UNKNOWN"
            if self.ctx.collector.get_record_count() > 0 and graceful_session:
                self._flush_collector(
                    session_id=str(graceful_session),
                    course_code=str(graceful_course),
                    reason="graceful shutdown",
                )
            self.logger.info("Attendance Pi client stopped.")


def main() -> None:
    PiAttendanceApp().run()


if __name__ == "__main__":
    main()
