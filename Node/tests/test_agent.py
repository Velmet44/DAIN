"""Node agent registration and graceful-shutdown tests."""

import asyncio
from types import SimpleNamespace

import dain_node.agent as agent_module
from dain_node.agent import NodeAgent, StopGuard


def test_stopguard_install_is_portable() -> None:
    async def install_guard() -> None:
        guard = StopGuard()
        guard.install()
        return guard

    guard = asyncio.run(install_guard())
    assert not guard.event.is_set()


def test_register_rejoins_when_persisted_token_is_stale(monkeypatch) -> None:
    class FakeIdentity:
        node_id = "node-1"
        node_token = "stale-node-token"

        def __init__(self) -> None:
            self.saved = False

        def save(self, _path: str) -> None:
            self.saved = True

    class FakeResponse:
        def __init__(self, body: dict) -> None:
            self.body = body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.body

    class FakeClient:
        def __init__(self) -> None:
            self.tokens: list[str] = []

        async def post(self, _url: str, *, json: dict) -> FakeResponse:
            self.tokens.append(json["auth_token"])
            if len(self.tokens) == 1:
                return FakeResponse({"accepted": False, "reason": "invalid join token"})
            return FakeResponse(
                {
                    "accepted": True,
                    "node_token": "fresh-node-token",
                    "heartbeat_interval_s": 5.0,
                }
            )

    async def run() -> tuple[FakeClient, FakeIdentity]:
        manifest = {
            "cpu": {"cores": 1, "ram_total_gb": 1.0, "ram_free_gb": 1.0},
            "net": {"bw_mbps": 1.0, "lat_ms_p95": 1.0},
        }
        monkeypatch.setattr(agent_module, "probe", lambda _settings: manifest)
        identity = FakeIdentity()
        client = FakeClient()
        agent = NodeAgent.__new__(NodeAgent)
        agent.settings = SimpleNamespace(
            join_token="dain-join",
            state_path="state.json",
            http_base_url="http://localhost:8000",
        )
        agent.handler = SimpleNamespace(store=SimpleNamespace(cached_inventory=lambda: ()))
        agent.peer_url = None
        agent._inventory_dirty = False
        agent._inventory = ()
        agent.identity = identity
        await agent.register(client)
        return client, identity

    client, identity = asyncio.run(run())

    assert client.tokens == ["stale-node-token", "dain-join"]
    assert identity.node_token == "fresh-node-token"
    assert identity.saved
