"""Admin access paths: localhost keyless, tailnet keyless (opt-in), public key."""

from starlette.requests import Request

from dain_coordinator.api import _admin_keyless_ok, _tailnet_trusted
from dain_coordinator.settings import CoordinatorSettings


def _request(
    peer: str,
    host: str = "daincoordinator.tailcae708.ts.net",
    origin: str | None = None,
) -> Request:
    headers = [(b"host", host.encode())]
    if origin:
        headers.append((b"origin", origin.encode()))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/admin/nodes",
            "query_string": b"",
            "client": (peer, 5000),
            "headers": headers,
        }
    )


def test_localhost_keyless_always_trusted() -> None:
    settings = CoordinatorSettings()
    assert _admin_keyless_ok(_request("127.0.0.1", host="127.0.0.1:8000"), settings)


def test_tailnet_peer_requires_opt_in() -> None:
    request = _request("100.90.15.24", origin="https://daincoordinator.tailcae708.ts.net")
    assert not _admin_keyless_ok(request, CoordinatorSettings())
    assert _tailnet_trusted(request)
    assert _admin_keyless_ok(
        request, CoordinatorSettings(admin_keyless_tailnet=True)
    )


def test_funnel_visitor_is_never_tailnet_trusted() -> None:
    # The funnel proxy connects from loopback: outside the tailnet range, so
    # public visitors always need the admin key regardless of the flag.
    assert not _tailnet_trusted(_request("127.0.0.1"))


def test_tailnet_peer_with_foreign_origin_rejected() -> None:
    # A drive-by page on a tailnet device fetching the coordinator: cross-site
    # POSTs carry Origin, which must match the tailnet/loopback allowlist.
    request = _request("100.90.15.24", origin="https://evil.example.com")
    assert not _tailnet_trusted(request)
    assert not _admin_keyless_ok(request, CoordinatorSettings(admin_keyless_tailnet=True))


def test_tailnet_peer_plain_ts_net_origin_trusted() -> None:
    request = _request("100.90.15.24", origin="http://daincoordinator.tailcae708.ts.net:8000")
    assert _tailnet_trusted(request)
