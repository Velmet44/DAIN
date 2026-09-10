"""Node package smoke test."""

import tomllib
from pathlib import Path


def test_node_version() -> None:
    import dain_node

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert dain_node.__version__ == expected
