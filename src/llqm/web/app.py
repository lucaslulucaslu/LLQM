"""FastAPI application factory."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from llqm.web.routes import home, investigation, partials, stream

_HERE = Path(__file__).resolve().parent


def create_app() -> FastAPI:
    app = FastAPI(title="LLQM Investigator")

    # Templates + static
    templates_dir = _HERE / "templates"
    static_dir = _HERE / "static"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routers
    app.include_router(home.router)
    app.include_router(investigation.router)
    app.include_router(stream.router)
    app.include_router(partials.router)

    return app


app = create_app()


def main() -> None:
    """Entry point for ``llqm-ui`` console script."""
    import uvicorn

    parser = argparse.ArgumentParser(description="LLQM web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "llqm.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
