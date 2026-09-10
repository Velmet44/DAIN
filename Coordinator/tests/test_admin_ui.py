"""Admin web UI: served shell + auth separation for its data feeds."""

from conftest import ADMIN_HEADERS
from fastapi.testclient import TestClient

from dain_coordinator.app import create_app
from dain_coordinator.settings import CoordinatorSettings


def _settings(tmp_path) -> CoordinatorSettings:
    return CoordinatorSettings(
        db_path=str(tmp_path / "ui.sqlite3"),
        admin_api_key="admin-ui-key",
        api_key="api-ui-key",
    )


def test_admin_ui_served(client: TestClient) -> None:
    response = client.get("/admin/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "DAIN Coordinator" in response.text
    assert "adminKey" in response.text and "apiKey" in response.text


def test_admin_ui_also_at_slashless_path(client: TestClient) -> None:
    assert client.get("/admin").status_code == 200


def test_ui_is_shell_not_authed(client: TestClient) -> None:
    """The shell loads without keys; the data endpoints stay locked."""
    assert client.get("/admin/").status_code == 200


def test_admin_api_still_locked(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/admin/nodes").status_code == 401
        assert client.get("/admin/nodes", headers=ADMIN_HEADERS).status_code == 401
        assert (
            client.get("/admin/nodes", headers={"X-Admin-Key": "admin-ui-key"}).status_code == 200
        )
        assert client.get("/ledger/summary").status_code == 401
        assert (
            client.get("/ledger/summary", headers={"X-API-Key": "api-ui-key"}).status_code == 200
        )
