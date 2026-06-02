"""Backend polling and active session state management."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, time
from typing import Any, Optional

from config import CONFIG_REFRESH_INTERVAL, STATE_FILE, STORAGE_DIR
from services.logger import get_logger, log_event


IDLE = "IDLE"
READY = "READY"
RECORDING = "RECORDING"
STOPPED = "STOPPED"


class SessionManager:
    def __init__(self, api_client) -> None:
        self.api_client = api_client
        self.logger = get_logger(__name__)
        self.refresh_interval = CONFIG_REFRESH_INTERVAL
        self._lock = threading.RLock()
        self._active_sessions: list[dict[str, Any]] = []
        self._last_sync_at: Optional[str] = None
        self._current_state: str = IDLE
        self._last_state_session_id: Optional[str] = None
        self._last_known_session: Optional[dict[str, Any]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def restore_state(self) -> None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        if not STATE_FILE.exists():
            STATE_FILE.write_text("{}", encoding="utf-8")
            return

        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.logger.warning("State file is corrupted; starting from a clean state.")
            return

        with self._lock:
            self._active_sessions = state.get("active_sessions", []) if isinstance(state, dict) else []
            self._last_sync_at = state.get("last_sync_at") if isinstance(state, dict) else None
            self._current_state = state.get("current_state", IDLE) if isinstance(state, dict) else IDLE
            stored_interval = state.get("refresh_interval") if isinstance(state, dict) else None
            if isinstance(stored_interval, int) and stored_interval > 0:
                self.refresh_interval = stored_interval
            self._last_known_session = state.get("last_known_session") if isinstance(state, dict) else None
            self._last_state_session_id = state.get("last_state_session_id") if isinstance(state, dict) else None

    def _save_state(self) -> None:
        state = {
            "active_sessions": self._active_sessions,
            "last_sync_at": self._last_sync_at,
            "refresh_interval": self.refresh_interval,
            "current_state": self._current_state,
            "last_known_session": self._last_known_session,
            "last_state_session_id": self._last_state_session_id,
        }
        tmp_path = STATE_FILE.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp_path.replace(STATE_FILE)

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        if value in (None, ""):
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_time(value: Any) -> Optional[time]:
        if value in (None, ""):
            return None
        if isinstance(value, time):
            return value
        if isinstance(value, datetime):
            return value.time()
        if isinstance(value, str):
            normalized = value.strip()
            if len(normalized) == 5:
                normalized = f"{normalized}:00"
            try:
                return time.fromisoformat(normalized)
            except ValueError:
                try:
                    return datetime.fromisoformat(normalized.replace("Z", "+00:00")).time()
                except ValueError:
                    return None
        return None

    def is_session_active_now(self, session: dict[str, Any] | None) -> bool:
        if not isinstance(session, dict):
            return False
        if str(session.get("status", "")).lower() != "active":
            return False

        session_date = self._parse_date(session.get("session_date"))
        start_time = self._parse_time(session.get("start_time"))
        end_time = self._parse_time(session.get("end_time"))
        if session_date is None or start_time is None or end_time is None:
            return False

        now = datetime.now()
        if now.date() != session_date:
            return False

        current_time = now.time()
        return start_time <= current_time <= end_time

    def _derive_state(self, session: Optional[dict[str, Any]]) -> str:
        if not session:
            if self._current_state in {RECORDING, READY} and self._last_known_session is not None:
                return STOPPED
            if self._current_state == STOPPED:
                return IDLE
            return IDLE

        status = str(session.get("status", "")).lower()
        if status != "active":
            return STOPPED if self._current_state in {RECORDING, READY} else IDLE

        if self.is_session_active_now(session):
            return RECORDING

        session_date = self._parse_date(session.get("session_date"))
        start_time = self._parse_time(session.get("start_time"))
        end_time = self._parse_time(session.get("end_time"))
        if session_date is None or start_time is None or end_time is None:
            return READY

        now = datetime.now()
        if now.date() != session_date:
            return READY if now.date() < session_date else STOPPED

        current_time = now.time()
        if current_time < start_time:
            return READY
        if start_time <= current_time <= end_time:
            return RECORDING
        return STOPPED

    def _update_state_locked(self) -> str:
        primary = self.get_primary_active_session()
        if primary is not None:
            self._last_known_session = primary

        new_state = self._derive_state(primary)
        previous_state = self._current_state
        previous_session_id = self._last_state_session_id
        current_session_id = str(primary.get("session_id")) if isinstance(primary, dict) and primary.get("session_id") is not None else None

        self._current_state = new_state
        self._last_state_session_id = current_session_id
        self._save_state()

        if new_state != previous_state:
            log_event(
                self.logger,
                "SESSION_STATE_CHANGE",
                previous=previous_state,
                current=new_state,
                session_id=current_session_id or previous_session_id or "UNKNOWN",
            )
            if new_state == RECORDING:
                log_event(
                    self.logger,
                    "SESSION_WINDOW_ACTIVE",
                    session_id=current_session_id or "UNKNOWN",
                )
            if previous_state == RECORDING and new_state in {READY, STOPPED, IDLE}:
                log_event(
                    self.logger,
                    "SESSION_WINDOW_INACTIVE",
                    previous=previous_state,
                    current=new_state,
                    session_id=previous_session_id or current_session_id or "UNKNOWN",
                )

        return self._current_state

    def refresh_sessions(self) -> None:
        payload = self.api_client.get_pi_config()
        if not payload or "active_sessions" not in payload:
            self.logger.warning("Backend config refresh failed; keeping previous session state.")
            return

        active_sessions = payload.get("active_sessions", [])
        refresh_interval = payload.get("refresh_interval")

        with self._lock:
            self._active_sessions = active_sessions if isinstance(active_sessions, list) else []
            if isinstance(refresh_interval, int) and refresh_interval > 0:
                self.refresh_interval = refresh_interval
            self._last_sync_at = datetime.now().isoformat()
            if self._active_sessions:
                self._last_known_session = self._active_sessions[0]
            self._save_state()

        self.logger.info(
            "Active sessions updated: %s session(s), refresh interval %ss.",
            len(self._active_sessions),
            self.refresh_interval,
        )

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.refresh_sessions()
                with self._lock:
                    self._update_state_locked()
            except Exception:
                self.logger.exception("Unhandled error while polling sessions.")
            self._stop_event.wait(self.refresh_interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="session-manager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_active_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._active_sessions)

    def get_primary_active_session(self) -> Optional[dict[str, Any]]:
        sessions = self.get_active_sessions()
        if not sessions:
            return None
        for session in sessions:
            if str(session.get("status", "")).lower() == "active":
                return session
        return sessions[0]

    def get_last_known_session(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._last_known_session) if isinstance(self._last_known_session, dict) else None

    def get_state(self) -> str:
        with self._lock:
            return self._update_state_locked()
