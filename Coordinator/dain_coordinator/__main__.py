"""Entry point: ``python -m dain_coordinator`` starts the coordinator API."""

import os

import uvicorn

from dain_coordinator.app import create_app


def main() -> None:
    port = int(os.environ.get("DAIN_PORT", "8000"))
    uvicorn.run(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
