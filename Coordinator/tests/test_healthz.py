"""S0 checkpoint: coordinator health endpoint responds on the API root."""

from fastapi.testclient import TestClient

from dain_coordinator.app import create_app


def test_healthz() -> None:
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
