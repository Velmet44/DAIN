"""Entry point: ``python -m dain_node`` runs the agent until SIGINT/SIGTERM."""

import asyncio
import logging

from dain_common.logging_setup import configure_logging

from dain_node.agent import run_agent
from dain_node.jobs import JobHandler
from dain_node.llm import ModelStoreClient
from dain_node.settings import NodeSettings


def main() -> int:
    settings = NodeSettings.from_env()
    configure_logging(json_mode=settings.log_json, level=logging.INFO)
    store = ModelStoreClient(cache_dir=settings.model_cache_dir)
    handler = JobHandler(settings, store)
    return asyncio.run(run_agent(settings, handler))


if __name__ == "__main__":
    raise SystemExit(main())
