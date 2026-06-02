"""Resilient backend API client for the Pi."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from config import (
    API_MAX_RETRIES,
    API_RETRY_BACKOFF_SECONDS,
    API_TIMEOUT_SECONDS,
    BACKEND_URL,
)
from services.logger import get_logger


class BackendApiClient:
    def __init__(
        self,
        backend_url: str = BACKEND_URL,
        timeout_seconds: int = API_TIMEOUT_SECONDS,
        max_retries: int = API_MAX_RETRIES,
        retry_backoff_seconds: int = API_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.logger = get_logger(__name__)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method: str, path: str, json_payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        url = f"{self.backend_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=json_payload,
                    timeout=self.timeout_seconds,
                )

                if response.status_code >= 500:
                    raise requests.RequestException(
                        f"Server error {response.status_code}: {response.text[:200]}"
                    )

                payload = self._safe_json(response)
                if response.status_code >= 400:
                    self.logger.warning(
                        "%s %s failed with status %s on attempt %s: %s",
                        method,
                        path,
                        response.status_code,
                        attempt,
                        payload,
                    )
                return {
                    "ok": response.status_code < 400,
                    "status_code": response.status_code,
                    "payload": payload,
                    "network_error": False,
                }
            except requests.RequestException as exc:
                last_error = exc
                self.logger.warning(
                    "Request attempt %s/%s failed for %s %s: %s",
                    attempt,
                    self.max_retries,
                    method,
                    path,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        self.logger.error("Request failed after %s attempts for %s %s: %s", self.max_retries, method, path, last_error)
        return {
            "ok": False,
            "status_code": None,
            "payload": {},
            "network_error": True,
            "error": str(last_error) if last_error else "unknown error",
        }

    @staticmethod
    def _safe_json(response: requests.Response) -> dict[str, Any]:
        try:
            data = response.json()
            return data if isinstance(data, dict) else {"data": data}
        except ValueError:
            return {"message": response.text}

    def get_pi_config(self) -> dict[str, Any]:
        response = self._request("GET", "/pi/config")
        payload = response.get("payload")
        if not isinstance(payload, dict):
            return {}
        return payload

    def get_pi_embeddings(self) -> dict[str, Any]:
        response = self._request("GET", "/pi/embeddings")
        payload = response.get("payload")
        if not isinstance(payload, dict):
            return {}
        return payload

    def post_attendance_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._request("POST", "/attendance/batch", json_payload=payload)
        response_payload = response.get("payload")
        if not isinstance(response_payload, dict):
            response_payload = {}
        response_payload["ok"] = bool(response.get("ok"))
        response_payload["status_code"] = response.get("status_code")
        response_payload["network_error"] = bool(response.get("network_error"))
        if response.get("network_error"):
            response_payload.setdefault("success", False)
            response_payload.setdefault("message", "Network failure")
        return response_payload
