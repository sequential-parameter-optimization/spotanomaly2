"""OAuth authentication for API access."""

import logging
import time
import urllib
from typing import Optional

import requests


class OAuthSession:
    """OAuth session for API authentication and requests."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        tech_user: str,
        tech_password: str,
        token_request_url: str,
        api_request_url: str,
        logger: Optional[logging.Logger] = None,
        verbose: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.tech_user = tech_user
        self.tech_password = tech_password
        self.token_request_url = token_request_url
        self.api_request_url = api_request_url
        self.refresh_token = None
        self.expires_at = 0
        self.access_token = None
        if not logger:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.get_access_token()

    def _request_token(self, body: dict) -> dict:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(self.token_request_url, headers=headers, data=body)
        response.raise_for_status()  # Handle errors in response
        return response.json()

    def _process_token_response(self, data: dict):
        self.expires_at = time.time() + data.get("expires_in", 3600) - 60  # Expire one minute early
        self.refresh_token = data["refresh_token"]
        self.access_token = data["access_token"]

    def get_access_token(self):
        body = {
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "grant_type": "password",
            "username": self.tech_user,
            "password": self.tech_password,
        }
        data = self._request_token(body)
        self._process_token_response(data)
        self.logger.info("Access token received")

    def refresh_access_token(self):
        body = {
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        data = self._request_token(body)
        self._process_token_response(data)
        self.logger.info("Access token refreshed")

    def make_request(self, path, params=None):
        if time.time() > self.expires_at:
            self.refresh_access_token()
        headers = {"Authorization": f"Bearer {self.access_token}"}
        path = urllib.parse.quote(path)
        r = requests.get(f"{self.api_request_url}{path}", headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json(), r.headers

    def fetch_all_pages(self, path, base_params=None, per_page=100, max_pages=10_000):
        """Fetch all pages, auto-handling several common pagination styles."""
        base_params = dict(base_params or {})
        results = []
        page = 1

        while page <= max_pages:
            params = {**base_params, "page": page, "per_page": per_page}
            data, headers = self.make_request(path, params=params)

            # ---- 1) Pure list response ----
            if isinstance(data, list):
                if not data:
                    break
                results.extend(data)
                if len(data) < per_page:
                    break
                page += 1
                continue

            # ---- 2) Envelope: {"data": [...], "meta": {"total_pages": N}} ----
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                batch = data["data"]
                results.extend(batch)
                total_pages = (data.get("meta") or {}).get("total_pages")
                if total_pages and page >= int(total_pages):
                    break
                if len(batch) < per_page and not total_pages:
                    break
                page += 1
                continue

            # ---- 3) Link header style (RFC 5988): Link: <...page=3>; rel="next" ----
            link = headers.get("Link") or headers.get("link")
            if link and 'rel="next"' in link:
                results.extend(data if isinstance(data, list) else data.get("data", []))
                page += 1
                continue

            # Fallback: if we can't recognize pagination, return what we have
            results.extend(data if isinstance(data, list) else data.get("data", []))
            break

        return results
