# squeeze_evolve

Core package source, organized into three subpackages.

```
squeeze_evolve/
  __init__.py            Public API: re-exports operator registries
  core/                  Infrastructure
    config.py            Pydantic config schema (RunConfig, ModelConfig, RoutingConfig)
    backend.py           OpenAI-compatible async backend with retry and batching
    data.py              Dataset loading (parquet, jsonl, json)
    registry.py          Generic decorator-based operator registry
  algorithm/             Evolutionary algorithm
    operators.py         Pluggable operators (fitness, selection, routing, recombination, etc.)
    orchestrator.py      Algorithm 1 loop, metrics dataclasses, checkpointing
  api/                   User interfaces
    cli.py               CLI entrypoints (squeeze-evolve-client, squeeze-evolve-server)
    server.py            FastAPI server (POST /run, GET /health)
```

## API server

```
GET  /health         -> {"status": "ok"}
GET  /capabilities   -> backend and transport info
POST /run            -> execute full evolutionary loop (RunConfig + problems)
```

## Metrics

Each loop emits structured metrics to `metrics_path` as flat JSON:

| Category | Fields |
|---|---|
| Routing | `model_{i}_count`, `lite_count`, `thresholds` |
| Tokens | `model_{i}_input_tokens`, `model_{i}_output_tokens`, `scoring_input_tokens`, `total_input_tokens`, `total_output_tokens` |
| Timing | `time_scoring_s`, `time_generation_s`, `time_total_s` |
| Confidence | `num_candidates_scored`, `mean/median/min/max/std_confidence` |
| Evaluation | `eval_*` (keys from the registered eval function, prefixed with `eval_`) |
