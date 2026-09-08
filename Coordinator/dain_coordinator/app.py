"""Coordinator ASGI application (S0 skeleton: health endpoint only).

Later stages mount: client REST+SSE (/v1/*), node WebSocket (/node/ws),
admin + metrics endpoints — see Docs/stages.md S2+.
"""

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="DAIN Coordinator", version="0.1.0")
    app.openapi_url = "/openapi.json"

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
