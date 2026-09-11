"""S9: in-memory sliding-window rate limiter for the public completion API.

Keyed by client IP (or API key) — one shared key inside a static SPA is wide
open to scraping, so a per-IP cap is the real guard on a public prototype
(spec §15/§17 posture). Disabled when the limit is 0 (tests/dev).
"""

from __future__ import annotations

import time


class RateLimiter:
    def __init__(self, limit_per_min: int = 0, max_keys: int = 10_000) -> None:
        self._limit = limit_per_min
        self._window_s = 60.0
        self._hits: dict[str, list[float]] = {}
        self._max_keys = max(1, max_keys)

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def _prune(self, now: float) -> None:
        """Drop keys that have gone idle for several windows so a burst of
        distinct clients can never grow `_hits` unboundedly."""
        if len(self._hits) <= self._max_keys:
            return
        cutoff = now - self._window_s * 4
        stale = [key for key, times in self._hits.items() if times[-1] < cutoff]
        for key in stale:
            del self._hits[key]

    def allow(self, key: str) -> bool:
        if not self.enabled:
            return True
        now = time.time()
        window = [t for t in self._hits.get(key, []) if t > now - self._window_s]
        if len(window) >= self._limit:
            self._hits[key] = window
            self._prune(now)
            return False
        window.append(now)
        self._hits[key] = window
        self._prune(now)
        return True
