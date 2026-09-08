"""Sim package smoke test."""


def test_sim_version() -> None:
    import dain_sim

    assert dain_sim.__version__ == "0.1.0"
