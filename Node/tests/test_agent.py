"""Node agent graceful-shutdown guard tests."""

import asyncio

from dain_node.agent import StopGuard


def test_stopguard_install_is_portable() -> None:
    async def install_guard() -> None:
        guard = StopGuard()
        guard.install()
        return guard

    guard = asyncio.run(install_guard())
    assert not guard.event.is_set()
