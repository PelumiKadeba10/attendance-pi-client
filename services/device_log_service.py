from __future__ import annotations

import queue
import threading
from datetime import datetime
from typing import Any, Optional

from services.logger import get_logger


class DeviceLogService:
    def __init__(self, api_client, device_id: str, max_queue_size: int = 256) -> None:
        self.api_client = api_client
        self.device_id = device_id
        self.logger = get_logger(__name__)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="device-log-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def log(self, event_type: str, message: str = "", metadata: Optional[dict[str, Any]] = None) -> None:
        payload = {
            "device_id": self.device_id,
            "event_type": event_type,
            "message": message,
        }
        if metadata is not None:
            payload["metadata"] = metadata

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            return

    def _send(self, payload: dict[str, Any]) -> None:
        try:
            response = self.api_client.post_device_log(payload)
            if not response.get("ok") and not response.get("network_error"):
                self.logger.debug("Device log rejected by backend: %s", response)
        except Exception:
            return

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._send(payload)
            self._queue.task_done()
