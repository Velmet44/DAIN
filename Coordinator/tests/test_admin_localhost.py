"""Keyless localhost admin (session 15).

The admin page on the same machine works without keys; remote hosts still need
them. The trust model requires ALL of: loopback socket peer, localhost Host
header (DNS-rebinding guard), and a localhost/absent Origin header (drive-by
CSRF guard — a visited web page always sends a cross-site Origin).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from dain_coordinator.app import create_app
from tests.conftest import ADMIN_KEY, API_KEY, make_settings


def client_for(
    tmp_path, *, peer: tuple[str, int] = ("127.0.0.1", 50000), base_url: str = "http://127.0.0.1:8000"
) -> TestClient:
    settings = make_settings(tmp_path, discovery_enabled=False)
    return TestClient(create_app(settings), client=peer, base_url=base_url)


def test_keyless_localhost_admin(tmp_path) -> None:
    with client_for(tmp_path) as client:
        assert client.get("/admin/nodes").status_code == 200
        assert client.post("/admin/models/rescan").status_code == 200
        # Secrets are readable locally — the whole point of a local console.
        keys = client.get("/admin/settings/keys").json()
        assert keys["admin_api_key"] == ADMIN_KEY
        # Keyless client API too (chat page + dashboard need no key locally).
        assert client.get("/v1/nodes").status_code == 200
        assert client.get("/v1/models").status_code == 200


def test_keyless_empty_header_counts_as_absent(tmp_path) -> None:
    with client_for(tmp_path) as client:
        r = client.get("/admin/nodes", headers={"X-Admin-Key": ""})
        assert r.status_code == 200


def test_dns_rebinding_blocked_by_host_check(tmp_path) -> None:
    # Socket peer is loopback but the browser resolves an attacker domain to
    # 127.0.0.1: the Host header is then evil.com -> must not be trusted.
    with client_for(tmp_path, base_url="http://evil.com:8000") as client:
        r = client.get("/admin/nodes")
        assert r.status_code == 401


def test_drive_by_origin_blocked(tmp_path) -> None:
    with client_for(tmp_path) as client:
        # A visited web page POSTing cross-origin always sends its Origin.
        r = client.post(
            "/admin/models/rescan", headers={"Origin": "https://evil.example"}
        )
        assert r.status_code == 401
        # The local SPA's own same-origin Origin is fine.
        r = client.post("/admin/models/rescan", headers={"Origin": "http://127.0.0.1:8000"})
        assert r.status_code == 200
        r = client.post("/admin/models/rescan", headers={"Origin": "http://localhost:5173"})
        assert r.status_code == 200


def test_remote_host_still_needs_keys(tmp_path) -> None:
    with client_for(tmp_path, peer=("192.0.2.5", 5000)) as client:
        assert client.get("/admin/nodes").status_code == 401
        assert client.get("/v1/nodes").status_code == 401
        # Keys keep working for remote admin/client access.
        assert client.get("/admin/nodes", headers={"X-Admin-Key": ADMIN_KEY}).status_code == 200
        assert client.get("/v1/nodes", headers={"X-API-Key": API_KEY}).status_code == 200
        # A wrong key from localhost is still rejected (explicit != keyless).
        with client_for(tmp_path) as local:
            r = local.get("/admin/nodes", headers={"X-Admin-Key": "wrong-key-1"})
            assert r.status_code == 401
