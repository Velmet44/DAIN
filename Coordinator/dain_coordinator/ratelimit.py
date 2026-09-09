"""S9: in-memory sliding-window rate limiter for the public completion API.

Keyed by client IP (or API key) — one shared key inside a static SPA is wide
open to scraping, so a per-IP cap is the real guard on a public prototype
(spec §15/§17 posture). Disabled when the limit is 0 (tests/dev).
"""

from __future__ import annotations

import time


class RateLimiter:
    def __init__(self, limit_per_min: int = 0) -> None:
        self._limit = limit_per_min
        self._window_s = 60.0
        self._hits: dict[str, list[float]] = {}

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def allow(self, key: str) -> bool:
        if not self.enabled:
            return True
        now = time.time()
        window = [t for t in self._hits.get(key, []) if t > now - self._window_s]
        if len(window) >= self._limit:
            self._hits[key] = window
            return False
        window.append(now)
        self._hits[key] = window
        return True
