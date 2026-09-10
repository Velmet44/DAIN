"""LAN coordinator discovery protocol (nodes never type an IP)."""

from dain_common.node_discovery import (
    DEFAULT_DISCOVERY_PORT,
    MAX_DATAGRAM_BYTES,
    DiscoverRequest,
    HelloReply,
    broadcast_targets,
    make_discover_packet,
    make_hello_packet,
    token_matches,
)


def test_default_port_is_valid() -> None:
    assert 1 <= DEFAULT_DISCOVERY_PORT <= 65535


def test_discover_packet_round_trip() -> None:
    packet = make_discover_packet("s3cret")
    req = DiscoverRequest.parse(packet)
    assert req is not None
    assert req.version == 1
    assert req.join_token == "s3cret"


def test_hello_packet_round_trip() -> None:
    reply = HelloReply.parse(make_hello_packet(8000))
    assert reply is not None
    assert reply.port == 8000


def test_parse_rejects_garbage() -> None:
    for blob in [b"", b"not json", b"{}", b'{"v":1,"op":"other"}', b"x" * (MAX_DATAGRAM_BYTES + 1)]:
        assert DiscoverRequest.parse(blob) is None
        assert HelloReply.parse(blob) is None


def test_parse_rejects_bad_token_and_port() -> None:
    assert DiscoverRequest.parse(b'{"v":1,"op":"discover","k":""}') is None
    assert DiscoverRequest.parse(b'{"v":1,"op":"discover"}') is None
    assert HelloReply.parse(b'{"v":1,"op":"hello","port":0}') is None
    assert HelloReply.parse(b'{"v":1,"op":"hello","port":70000}') is None


def test_token_matches() -> None:
    assert token_matches("abc", "abc")
    assert not token_matches("abc", "abd")
    assert not token_matches("abc", "abcd")


def test_broadcast_targets_covers_subnet() -> None:
    targets = broadcast_targets()
    assert "255.255.255.255" in targets
