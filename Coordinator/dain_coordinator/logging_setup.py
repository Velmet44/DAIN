"""Logging setup with optional JSON output and node/job correlation.

Correlation convention: domain log calls pass ``node_id`` / ``job_id`` via the
standard ``extra`` mapping; the JSON formatter lifts them to top-level keys, and
domain messages repeat the node_id in text so the default console format stays
greppable.
"""

from __future__ import annotations

import json
import logging
import sys

_NAMESPACE = "dain"
_EXTRA_KEYS = ("node_id", "job_id", "seq", "from_state", "to_state", "reason", "event")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in _EXTRA_KEYS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(json_mode: bool = False, level: int = logging.INFO) -> None:
    """Idempotent: safe to call on every create_app()."""
    logger = logging.getLogger(_NAMESPACE)
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    if json_mode:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
