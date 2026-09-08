"""Entry point: ``python -m dain_coordinator`` starts the coordinator API."""

import uvicorn

from dain_coordinator.app import create_app
from dain_coordinator.settings import CoordinatorSettings


def main() -> None:
    settings = CoordinatorSettings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
