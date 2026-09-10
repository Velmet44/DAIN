"""Entry point: ``python -m dain_coordinator`` starts the coordinator API.

Settings come from ``Coordinator/config.json`` (a full default is written on
first run) with environment variables winning per-field; see
``dain_coordinator.config.resolve_settings``.
"""

import logging
import socket

import uvicorn

from dain_coordinator.app import create_app
from dain_coordinator.config import resolve_settings


def _find_open_port(host: str, preferred: int) -> int:
    """Return *preferred* if free, else scan nearby ports."""
    for port in [preferred, preferred + 1, preferred + 2, 8080, 8888, 9000]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found near {preferred}")


def main() -> None:
    settings, path, created = resolve_settings()
    log = logging.getLogger("dain.coordinator")
    log.info(
        "coordinator_start host=%s port=%d config=%s config_created=%s",
        settings.host,
        settings.port,
        path,
        created,
    )
    port = _find_open_port(settings.host, settings.port)
    if port != settings.port:
        print(f"[DAIN] Port {settings.port} busy — using {port}")
    uvicorn.run(create_app(settings), host=settings.host, port=port)


if __name__ == "__main__":
    main()
