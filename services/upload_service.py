"""Upload manager for cached and live attendance batches."""

from __future__ import annotations

import time
from typing import Any

from services.cache_service import load_pending_batches, remove_uploaded_batch, save_pending_batch
from services.logger import get_logger, log_event


class UploadService:
    def __init__(self, api_client) -> None:
        self.api_client = api_client
        self.logger = get_logger(__name__)

    def reload_cache(self) -> list[dict[str, Any]]:
        batches = load_pending_batches()
        log_event(self.logger, "CACHE_RELOADED", count=len(batches))
        return batches

    def _post_with_backoff(self, batch: dict[str, Any], label: str) -> dict[str, Any]:
        max_attempts = 4
        backoff_seconds = 1.0
        response: dict[str, Any] = {"ok": False, "network_error": True}

        for attempt in range(1, max_attempts + 1):
            response = self.api_client.post_attendance_batch(batch)
            if response.get("ok"):
                return response

            if not response.get("network_error"):
                return response

            if attempt < max_attempts:
                log_event(
                    self.logger,
                    "UPLOAD_RETRY",
                    label=label,
                    attempt=attempt,
                    backoff_seconds=backoff_seconds,
                )
                time.sleep(backoff_seconds)
                backoff_seconds *= 2

        return response

    def upload_cached_batches(self) -> bool:
        cached_batches = load_pending_batches()
        if not cached_batches:
            return True

        for batch in cached_batches:
            cache_id = str(batch.get("cache_id"))
            log_event(self.logger, "UPLOAD_CACHED_BATCH", cache_id=cache_id)
            response = self._post_with_backoff(batch, label=f"cached:{cache_id}")
            if response.get("ok"):
                remove_uploaded_batch(cache_id)
                continue

            if response.get("network_error"):
                self.logger.warning("Cached batch %s upload failed because of a network error: %s", cache_id, response)
            else:
                self.logger.error("Cached batch %s was rejected by the backend: %s", cache_id, response)
            return False

        return True

    def upload_current_batch(self, batch: dict[str, Any]) -> bool:
        log_event(self.logger, "UPLOAD_CURRENT_BATCH", session_id=batch.get("session_id") or "UNKNOWN")
        response = self._post_with_backoff(batch, label="current")
        if response.get("ok"):
            return True

        if response.get("network_error"):
            self.logger.warning("Current batch upload failed after retries; caching batch for later retry.")
            save_pending_batch(batch)
        else:
            self.logger.error("Current batch was rejected by the backend and will not be cached automatically: %s", response)
        return False

    def upload_cycle(self, batch: dict[str, Any]) -> bool:
        cached_ok = self.upload_cached_batches()
        current_ok = self.upload_current_batch(batch)
        return cached_ok and current_ok
