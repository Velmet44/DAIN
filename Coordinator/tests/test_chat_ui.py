"""Chat web UI (/msg): served shell + auth separation for its data feeds."""

from fastapi.testclient import TestClient

from dain_coordinator.app import create_app
from dain_coordinator.settings import CoordinatorSettings


def _settings(tmp_path) -> CoordinatorSettings:
    return CoordinatorSettings(
        db_path=str(tmp_path / "ui.sqlite3"),
        admin_api_key="admin-ui-key",
        api_key="api-ui-key",
    )


def test_chat_ui_served(client: TestClient) -> None:
    response = client.get("/msg/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "DAIN Chat" in response.text
    assert 'id="prompt"' in response.text and "/v1/completions" in response.text


def test_chat_ui_also_at_slashless_path(client: TestClient) -> None:
    assert client.get("/msg").status_code == 200


def test_chat_ui_is_shell_not_authed(client: TestClient) -> None:
    """The shell loads without keys; the data endpoints stay locked."""
    assert client.get("/msg/").status_code == 200


def test_chat_data_feeds_stay_locked(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get("/v1/models", headers={"X-API-Key": "api-ui-key"}).status_code == 200
        )
        assert (
            client.post(
                "/v1/completions", json={"model_id": "x", "prompt": "hi"}
            ).status_code
            == 401
        )
