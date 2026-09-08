"""Entry point: ``python -m dain_node`` runs the agent until SIGINT/SIGTERM."""

import asyncio

from dain_node.agent import run_agent
from dain_node.executor import FakeExecutor
from dain_node.settings import NodeSettings


def main() -> int:
    settings = NodeSettings.from_env()
    # S3 ships the deterministic FakeExecutor; model executors are wired in S4.
    return asyncio.run(run_agent(settings, FakeExecutor()))


if __name__ == "__main__":
    raise SystemExit(main())
