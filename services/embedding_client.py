"""Backend-synced embedding cache for the Pi recognizer."""

from __future__ import annotations

import threading
from typing import Optional

from config import EMBEDDING_REFRESH_INTERVAL
from recognition.embedding_loader import EmbeddingIndex
from services.logger import get_logger, log_event


class EmbeddingClient:
    def __init__(self, api_client, refresh_interval: int = EMBEDDING_REFRESH_INTERVAL) -> None:
        self.api_client = api_client
        self.refresh_interval = refresh_interval
        self.logger = get_logger(__name__)
        self.embedding_index = EmbeddingIndex()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._last_version: Optional[str] = None

    def refresh_embeddings(self) -> bool:
        payload = self.api_client.get_pi_embeddings()
        if not isinstance(payload, dict):
            return False

        if payload.get("error") == "embedding_fetch_failed":
            self.logger.warning("Embedding refresh failed on backend; retaining existing cache.")
            return False

        embeddings = payload.get("embeddings", [])
        version = payload.get("version")
        if not isinstance(embeddings, list):
            return False

        updated = self.embedding_index.update_from_records(embeddings, version=version)
        with self._lock:
            self._last_version = version if isinstance(version, str) else None

        log_event(
            self.logger,
            "EMBEDDINGS_REFRESHED",
            count=updated,
            version=self._last_version or "UNKNOWN",
        )
        return True

    def _refresh_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.refresh_embeddings()
            except Exception:
                self.logger.exception("Unhandled error while refreshing embeddings.")
            self._stop_event.wait(self.refresh_interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._refresh_loop, name="embedding-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_embedding_index(self) -> EmbeddingIndex:
        return self.embedding_index

    def get_version(self) -> Optional[str]:
        with self._lock:
            return self._last_version
