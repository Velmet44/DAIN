"""Node package smoke test."""


def test_node_version() -> None:
    import dain_node

    assert dain_node.__version__ == "0.1.0"
