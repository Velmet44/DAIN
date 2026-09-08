"""S0 smoke test: the shared package is importable and versioned."""

import dain_common


def test_common_version() -> None:
    assert dain_common.__version__ == "0.1.0"
