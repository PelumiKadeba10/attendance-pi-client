"""In-memory embedding index used by the Pi recognizer."""

from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np


class EmbeddingIndex:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._embeddings: dict[str, np.ndarray] = {}
        self._version: Optional[str] = None

    @staticmethod
    def _normalize_vector(vector: Any) -> Optional[np.ndarray]:
        if vector is None:
            return None

        try:
            normalized = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError):
            return None

        if normalized.size == 0:
            return None
        return normalized

    def update_from_records(self, records: list[dict[str, Any]], version: Optional[str] = None) -> int:
        embeddings: dict[str, np.ndarray] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            student_id = record.get("student_id")
            vector = self._normalize_vector(record.get("embedding"))
            if student_id is None or vector is None:
                continue
            embeddings[str(student_id)] = vector

        with self._lock:
            self._embeddings = embeddings
            self._version = version

        return len(embeddings)

    def snapshot(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {student_id: vector.copy() for student_id, vector in self._embeddings.items()}

    def match(self, query_embedding: np.ndarray, threshold: float) -> tuple[Optional[str], float]:
        with self._lock:
            embeddings = dict(self._embeddings)

        if query_embedding is None or not embeddings:
            return None, 0.0

        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return None, 0.0

        best_student_id: Optional[str] = None
        best_score = -1.0

        for student_id, known_embedding in embeddings.items():
            candidate = np.asarray(known_embedding, dtype=np.float32)
            candidate_norm = np.linalg.norm(candidate)
            if candidate_norm == 0.0:
                continue
            score = float(np.dot(query, candidate) / (query_norm * candidate_norm))
            if score > best_score:
                best_student_id = student_id
                best_score = score

        if best_score < threshold:
            return None, best_score
        return best_student_id, best_score

