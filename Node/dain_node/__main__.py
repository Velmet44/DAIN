"""Entry point: ``python -m dain_node`` runs the agent until SIGINT/SIGTERM."""

import asyncio

from dain_node.agent import run_agent
from dain_node.jobs import JobHandler
from dain_node.llm import ModelStoreClient
from dain_node.settings import NodeSettings


def main() -> int:
    settings = NodeSettings.from_env()
    store = ModelStoreClient(cache_dir=settings.model_cache_dir)
    handler = JobHandler(settings, store)
    return asyncio.run(run_agent(settings, handler))


if __name__ == "__main__":
    raise SystemExit(main())
