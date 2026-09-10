"""In-memory log ring for the admin log viewer.

Attached to the root logger once per process; keeps the last ``CAPACITY``
formatted lines so the admin UI can tail coordinator activity without disk
access or external log shipping.
"""

from __future__ import annotations

import collections
import logging

CAPACITY = 2000

ring_logger = logging.getLogger("dain.admin.logs")


class RingLogHandler(logging.Handler):
    """Format + keep the last ``capacity`` records in memory."""

    def __init__(self, capacity: int = CAPACITY) -> None:
        super().__init__()
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
        )
        self._events: collections.deque[str] = collections.deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._events.append(self.format(record))
        except Exception:
            self.handleError(record)

    def snapshot(self, lines: int) -> list[str]:
        events = list(self._events)
        return events[-lines:]


_RING: RingLogHandler | None = None


def attach_ring(level: int = logging.INFO) -> RingLogHandler:
    """Attach the shared ring to the root logger exactly once per process."""
    global _RING
    if _RING is None:
        _RING = RingLogHandler()
        _RING.setLevel(level)
        root = logging.getLogger()
        if _RING not in root.handlers:
            root.addHandler(_RING)
            ring_logger.info("log_ring_attached capacity=%d", CAPACITY)
    return _RING


def snapshot(lines: int) -> list[str]:
    return attach_ring().snapshot(lines)
