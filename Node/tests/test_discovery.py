"""Node LAN discovery: probe against a fake coordinator over loopback."""

import asyncio
import json
import socket
from contextlib import suppress
from pathlib import Path

from dain_common.node_discovery import DiscoverRequest, make_hello_packet, token_matches

from dain_node.config import find_base_dir, load_config, write_config
from dain_node.discovery import discover_coordinator
from dain_node.settings import NodeSettings


def _make_responder(join_token: str, ws_port: int) -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    port = sock.getsockname()[1]
    return sock, port


async def _serve_responder(
    sock: socket.socket, join_token: str, ws_port: int
) -> None:
    """Answer probes that authenticate with *join_token*; ignore everything else."""
    loop = asyncio.get_running_loop()
    reply = make_hello_packet(ws_port)
    while True:
        try:
            data, addr = await loop.sock_recvfrom(sock, 512)
        except OSError:
            return
        req = DiscoverRequest.parse(data)
        if req is not None and token_matches(join_token, req.join_token):
            await loop.sock_sendto(sock, reply, addr)


def _run_probe(join_token: str, ws_port: int, probe_token: str) -> str | None:
    async def scenario() -> str | None:
        sock, port = _make_responder(join_token, ws_port)
        task = asyncio.create_task(_serve_responder(sock, join_token, ws_port))
        try:
            return await discover_coordinator(
                probe_token, port=port, targets=["127.0.0.1"], timeout_s=1.0
            )
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            sock.close()

    return asyncio.run(scenario())


def test_discovers_coordinator_and_builds_ws_url() -> None:
    url = _run_probe(join_token="s3cret", ws_port=8123, probe_token="s3cret")
    assert url == "ws://127.0.0.1:8123"


def test_wrong_token_gets_no_reply() -> None:
    assert _run_probe("real", 8123, "wiretap") is None


def test_no_responder_fails_cleanly() -> None:
    sock, port = _make_responder("x", 1)
    sock.close()

    async def probe() -> str | None:
        return await discover_coordinator(
            "x", port=port, targets=["127.0.0.1"], timeout_s=0.5
        )

    assert asyncio.run(probe()) is None


def test_config_write_then_load_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    data = {"coord_url": "ws://192.168.1.10:8000", "join_token": "x"}
    write_config(config_path, data)
    assert load_config(config_path) == data
    assert json.loads(config_path.read_text(encoding="utf-8")) == data


def test_from_config_blank_coord_url_falls_back_to_localhost(tmp_path: Path) -> None:
    settings = NodeSettings.from_config({"coord_url": ""}, find_base_dir())
    assert settings.coord_url == "ws://localhost:8000"
