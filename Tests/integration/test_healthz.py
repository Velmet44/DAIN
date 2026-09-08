"""S0 checkpoint: coordinator health endpoint responds on the API root."""

from dain_coordinator.app import create_app
from fastapi.testclient import TestClient


def test_healthz() -> None:
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
