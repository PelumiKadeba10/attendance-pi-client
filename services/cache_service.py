"""Persistent cache for attendance batches that need retrying."""

from __future__ import annotations

import json
import hashlib
import threading
import uuid
from datetime import datetime
from typing import Any

from config import PENDING_BATCHES_FILE, STORAGE_DIR
from services.logger import get_logger

_LOCK = threading.Lock()


def _ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not PENDING_BATCHES_FILE.exists():
        PENDING_BATCHES_FILE.write_text("[]", encoding="utf-8")


def _read_pending_batches() -> list[dict[str, Any]]:
    _ensure_storage()
    try:
        data = json.loads(PENDING_BATCHES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _write_pending_batches(batches: list[dict[str, Any]]) -> None:
    _ensure_storage()
    tmp_path = PENDING_BATCHES_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(batches, indent=2), encoding="utf-8")
    tmp_path.replace(PENDING_BATCHES_FILE)


def _batch_signature(batch: dict[str, Any]) -> str:
    normalized = {
        "session_id": batch.get("session_id"),
        "device_id": batch.get("device_id"),
        "records": batch.get("records", []),
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_pending_batch(batch: dict[str, Any]) -> str:
    logger = get_logger(__name__)
    with _LOCK:
        batches = _read_pending_batches()
        signature = _batch_signature(batch)
        for existing in batches:
            if existing.get("batch_signature") == signature:
                return str(existing.get("cache_id"))
        cache_id = str(uuid.uuid4())
        stored = dict(batch)
        stored["cache_id"] = cache_id
        stored["batch_signature"] = signature
        stored["queued_at"] = stored.get("queued_at") or datetime.now().isoformat()
        batches.append(stored)
        _write_pending_batches(batches)
        logger.info("Saved pending batch %s with %s records.", cache_id, len(stored.get("records", [])))
        return cache_id


def load_pending_batches() -> list[dict[str, Any]]:
    with _LOCK:
        batches = _read_pending_batches()
        return sorted(batches, key=lambda batch: str(batch.get("queued_at") or ""))


def remove_uploaded_batch(cache_id: str) -> None:
    logger = get_logger(__name__)
    with _LOCK:
        batches = _read_pending_batches()
        retained = [batch for batch in batches if str(batch.get("cache_id")) != str(cache_id)]
        if len(retained) != len(batches):
            _write_pending_batches(retained)
            logger.info("Removed uploaded cached batch %s.", cache_id)
