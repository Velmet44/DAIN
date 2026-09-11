"""S2 checkpoint: 20 concurrent fake nodes, 60 s, zero spurious state flips.

This is the real-socket stability test: a genuine coordinator (uvicorn, in
process) plus 20 genuine WS clients at production timing (5 s heartbeats).
"""

import asyncio

import httpx
from dain_coordinator.settings import CoordinatorSettings
from helpers import (
    ADMIN_HEADERS,
    ADMIN_KEY,
    JOIN_TOKEN,
    cpu_only_manifest,
    fake_node,
    gpu_manifest,
    register_node,
)

from dain_sim.server import ClusterServer, start_server, stop_server

N_NODES = 20
DURATION_S = 60.0
HEARTBEAT_S = 5.0


def test_twenty_nodes_sixty_seconds_stable(tmp_path) -> None:
    async def main() -> None:
        settings = CoordinatorSettings(
            db_path=str(tmp_path / "coordinator.sqlite3"),
            admin_api_key=ADMIN_KEY,
            join_token=JOIN_TOKEN,
        )  # production timing
        cluster: ClusterServer = await start_server(settings)
        tokens: dict[str, str] = {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Mixed heterogeneous pool: 8 high-end GPUs, 8 mid GPUs, 4 CPU-only.
                manifests = {}
                for i in range(N_NODES):
                    if i < 8:
                        manifests[f"node-{i:02d}"] = gpu_manifest("RTX 4090", 165.0, 24.0, 450.0)
                    elif i < 16:
                        manifests[f"node-{i:02d}"] = gpu_manifest("RTX 3060", 51.0, 12.0, 170.0)
                    else:
                        manifests[f"node-{i:02d}"] = cpu_only_manifest()

                acks = await asyncio.gather(
                    *(
                        register_node(client, cluster.base_url, node_id, manifest)
                        for node_id, manifest in manifests.items()
                    )
                )
                for node_id, ack in zip(manifests, acks, strict=True):
                    assert ack["accepted"] is True, (node_id, ack)
                    tokens[node_id] = ack["node_token"]

                # 60 s of heartbeats, staggered starts to avoid a thundering herd.
                sent = await asyncio.gather(
                    *(
                        fake_node(
                            cluster.ws_url,
                            node_id,
                            token,
                            heartbeat_interval_s=HEARTBEAT_S,
                            duration_s=DURATION_S,
                            stagger_s=i * 0.1,
                        )
                        for i, (node_id, token) in enumerate(tokens.items())
                    )
                )

                listing = (
                    await client.get(
                        f"{cluster.base_url}/admin/nodes", headers=ADMIN_HEADERS
                    )
                ).json()
                assert len(listing) == N_NODES
                assert all(n["state"] == "online" for n in listing), listing

                for node_id in tokens:
                    detail = (
                        await client.get(
                            f"{cluster.base_url}/admin/nodes/{node_id}",
                            headers=ADMIN_HEADERS,
                        )
                    ).json()
                    assert detail["state"] == "online", (node_id, detail)
                    # Exactly one transition each (the registration): no spurious flips.
                    assert len(detail["history"]) == 1, (node_id, detail["history"])
                    assert detail["history"][0]["to_state"] == "online"

                assert all(count >= DURATION_S / HEARTBEAT_S - 2 for count in sent), sent
        finally:
            await stop_server(cluster)

    asyncio.run(main())
