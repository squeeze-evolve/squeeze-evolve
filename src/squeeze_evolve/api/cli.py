"""CLI entrypoints for client and server."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
from pathlib import Path


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _discover_benchmarks() -> None:
    """Auto-discover and execute all benchmarks/*/register.py files."""
    bench_dir = Path(__file__).resolve().parents[3] / "benchmarks"
    if not bench_dir.is_dir():
        return
    for reg in sorted(bench_dir.glob("*/register.py")):
        spec = importlib.util.spec_from_file_location(
            f"_bench_{reg.parent.name}", str(reg),
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)


def client() -> None:
    from ..core.data import load_dataset
    from ..algorithm.orchestrator import RoutingOrchestrator
    from ..algorithm.utils import load_run_config

    parser = argparse.ArgumentParser(description="Run SqueezeEvolve client.")
    parser.add_argument("--config", required=True, help="Registry name, .py/.yaml/.json file")
    parser.add_argument("--input", required=True, help="Problems file (.json, .parquet, .jsonl)")
    parser.add_argument("--output", default="")
    parser.add_argument("--n-problems", type=int, default=None, help="Limit number of problems loaded")
    parser.add_argument("--include-path", default=None, help="Directory of .py config presets to discover")
    args = parser.parse_args()

    _configure_logging()
    _discover_benchmarks()

    cfg = load_run_config(args.config, include_path=args.include_path)
    problems = load_dataset(args.input, n_problems=args.n_problems, multimodal=cfg.routing.multimodal)

    result = asyncio.run(RoutingOrchestrator(cfg).run(problems))
    from dataclasses import asdict, is_dataclass

    def _default(obj):
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    rendered = json.dumps(result, indent=2, default=_default)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(rendered)
    else:
        print(rendered)


def serve() -> None:
    import uvicorn

    _discover_benchmarks()

    parser = argparse.ArgumentParser(description="Run SqueezeEvolve server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run("squeeze_evolve.api.server:app", host=args.host, port=args.port, reload=False)
