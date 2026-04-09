"""Shared test fixtures — auto-discover benchmark registrations."""

import importlib.util
from pathlib import Path

# Ensure the local squeeze_evolve is loaded into sys.modules before
# benchmark register.py files try to import it via importlib.
import squeeze_evolve  # noqa: F401
import squeeze_evolve.algorithm.operators  # noqa: F401


def _discover_benchmarks() -> None:
    bench_dir = Path(__file__).resolve().parents[1] / "benchmarks"
    if not bench_dir.is_dir():
        return
    for reg in sorted(bench_dir.glob("*/register.py")):
        spec = importlib.util.spec_from_file_location(
            f"_bench_{reg.parent.name}", str(reg),
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)


_discover_benchmarks()
