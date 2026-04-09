"""FastAPI server for orchestration jobs."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from ..core.config import RunConfig
from ..algorithm.orchestrator import RoutingOrchestrator

app = FastAPI(title="SqueezeEvolve", version="0.1.0")


class RunRequest(BaseModel):
    config: RunConfig
    problems: list[dict[str, Any]]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    return {
        "transport": "async-openai-sdk-only",
        "backends": ["closed_api", "openweight_api", "local_vllm"],
    }


@app.post("/run")
async def run(req: RunRequest) -> dict[str, Any]:
    return await RoutingOrchestrator(req.config).run(req.problems)
