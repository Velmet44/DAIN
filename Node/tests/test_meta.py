"""S23 meta-bootstrap: coord_url extraction + settings plumbing."""

from pathlib import Path

from dain_node.agent import _meta_coord_url
from dain_node.settings import NodeSettings


def test_meta_url_normalizes_browser_schemes() -> None:
    assert _meta_coord_url({"coord_url": "https://box.tail-scale.ts.net"}) == (
        "wss://box.tail-scale.ts.net"
    )
    assert _meta_coord_url({"coord_url": "http://192.168.1.5:8000"}) == (
        "ws://192.168.1.5:8000"
    )


def test_meta_url_garbage_is_ignored() -> None:
    assert _meta_coord_url({}) is None
    assert _meta_coord_url({"coord_url": ""}) is None
    assert _meta_coord_url({"coord_url": "   "}) is None
    assert _meta_coord_url({"coord_url": "ftp://nope"}) is None


def test_settings_meta_url_from_env_and_config() -> None:
    import os

    os.environ["DAIN_META_URL"] = "https://example.test/meta.json"
    try:
        assert NodeSettings.from_env().meta_url == "https://example.test/meta.json"
    finally:
        del os.environ["DAIN_META_URL"]
    cfg = NodeSettings.from_config({"meta_url": "https://cfg.test/meta.json"}, Path("."))
    assert cfg.meta_url == "https://cfg.test/meta.json"
    assert NodeSettings.from_config({}, Path(".")).meta_url is None
