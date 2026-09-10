"""LAN discovery responder: real UDP round-trip over loopback."""

import asyncio
import socket
from contextlib import suppress
from typing import Any

import pytest
from dain_common.node_discovery import make_discover_packet

from dain_coordinator.discovery import DiscoveryResponder


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _probe(port: int, token: str, timeout: float = 0.5) -> bytes | None:
    """Send one discover probe to the responder; return the reply bytes."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    try:
        loop = asyncio.get_running_loop()
        await loop.sock_sendto(sock, make_discover_packet(token), ("127.0.0.1", port))
        data, _ = await asyncio.wait_for(loop.sock_recvfrom(sock, 512), timeout)
        return data
    except TimeoutError:
        return None
    finally:
        sock.close()


def test_responder_answers_authenticated_probe() -> None:
    port = _free_udp_port()
    responder = DiscoveryResponder(port, "join-token", 8000)
    responder.start()
    try:

        async def scenario() -> bytes | None:
            task = asyncio.create_task(responder.run())
            try:
                return await _probe(port, "join-token")
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        reply = asyncio.run(scenario())
        assert reply == b'{"v":1,"op":"hello","port":8000}'
    finally:
        responder.close()


def test_responder_silent_on_wrong_token() -> None:
    port = _free_udp_port()
    responder = DiscoveryResponder(port, "real-token", 8001)
    responder.start()
    try:

        async def scenario() -> bytes | None:
            task = asyncio.create_task(responder.run())
            try:
                return await _probe(port, "wrong-token")
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        assert asyncio.run(scenario()) is None
    finally:
        responder.close()


def test_responder_bind_failure_raises_oserror() -> None:
    port = _free_udp_port()
    first = DiscoveryResponder(port, "t", 8000)
    first.start()
    try:
        second = DiscoveryResponder(port, "t", 8000)
        with pytest.raises(OSError):
            second.start()
    finally:
        first.close()


def test_settings_discovery_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from dain_coordinator.settings import CoordinatorSettings

    monkeypatch.delenv("DAIN_DISCOVERY_ENABLED", raising=False)
    assert CoordinatorSettings.from_env().discovery_enabled is True
    monkeypatch.setenv("DAIN_DISCOVERY_ENABLED", "false")
    monkeypatch.setenv("DAIN_DISCOVERY_PORT", "9999")
    settings: Any = CoordinatorSettings.from_env()
    assert settings.discovery_enabled is False
    assert settings.discovery_port == 9999
