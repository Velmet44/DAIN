"""Unit tests for the simulator entrypoints (no sockets, no subprocess runs).

Pins the exact environment a spawned agent receives, platform-specific spawn
flags, graceful-stop behaviour on an already-dead process, and the
`chaos --kill-at TIME:NODE` grammar that previously parsed but was ignored.
"""

import os
import subprocess
import sys

import pytest

from dain_sim.chaos import _parse_kill_at
from dain_sim.cluster import _graceful_stop, _node_env, _spawn_flags
from dain_sim.dev import JOIN_TOKEN
from dain_node.shard_export import DEV_MODEL_ID


def test_spawn_flags_platform_consistent() -> None:
    flags = _spawn_flags()
    if os.name == "nt":
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert flags == 0


def test_node_env_carries_endpoint_and_credentials() -> None:
    workdir = os.path.join("tmp", "w") if os.name != "nt" else r"C:\tmp\w"
    env = _node_env(port=8123, node_id="node-02", workdir=workdir, heartbeat_s=1.5, idx=2)
    assert env["DAIN_COORD_URL"] == "ws://127.0.0.1:8123"
    assert env["DAIN_JOIN_TOKEN"] == JOIN_TOKEN
    assert env["DAIN_NODE_ID"] == "node-02"
    assert env["DAIN_HEARTBEAT_S"] == "1.5"
    assert env["DAIN_MODEL"] == DEV_MODEL_ID
    assert env["DAIN_NET_BW_MBPS"] == "200"  # 150 + idx*25 (score differentiation)
    assert os.path.basename(env["DAIN_NODE_STATE_PATH"]) == "node_state.json"


def test_graceful_stop_is_noop_for_dead_process() -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _graceful_stop(dead)  # must not raise even though poll() is not None


@pytest.mark.parametrize(
    ("spec", "expected_s", "expected_target"),
    [
        ("1s:node-03", 1.0, "node-03"),
        ("500ms:sampling", 0.5, "sampling"),
        ("2m:node-00", 120.0, "node-00"),
        ("30s:node-01", 30.0, "node-01"),
    ],
)
def test_parse_kill_at_grammar(spec: str, expected_s: float, expected_target: str) -> None:
    got_s, got_target = _parse_kill_at(spec)
    assert got_s == pytest.approx(expected_s)
    assert got_target == expected_target


def test_parse_kill_at_rejects_bad_time() -> None:
    with pytest.raises(ValueError):
        _parse_kill_at("soon:node-01")