"""S3: identity persistence — stable node_id + issued token across restarts."""

import json

from dain_node.identity import IdentityState, generate_node_id


def test_fresh_identity_persists_id_only(tmp_path) -> None:
    path = str(tmp_path / "state.json")
    state = IdentityState.create(path, node_id="node-fixed")
    assert state.node_id == "node-fixed"
    assert state.node_token is None

    loaded = IdentityState.load(path)
    assert loaded is not None
    assert loaded.node_id == "node-fixed" and loaded.node_token is None


def test_token_persistence_round_trip(tmp_path) -> None:
    path = str(tmp_path / "state.json")
    state = IdentityState.create(path)
    state.node_token = "tok_abcdef123456"
    state.save(path)

    loaded = IdentityState.load(path)
    assert loaded is not None
    assert loaded.node_id == state.node_id
    assert loaded.node_token == "tok_abcdef123456"


def test_corrupt_state_starts_fresh(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    assert IdentityState.load(str(path)) is None


def test_generated_node_ids_are_stable_format() -> None:
    first, second = generate_node_id(), generate_node_id()
    assert first.startswith("node-") and second.startswith("node-")
    assert first != second  # random suffix


def test_save_is_atomic_no_tmp_leftovers(tmp_path) -> None:
    path = str(tmp_path / "state.json")
    state = IdentityState.create(path)
    state.node_token = "tok_x"
    state.save(path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith("state.json.tmp")]
    assert leftovers == []
    assert (
        json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["node_token"] == "tok_x"
    )
