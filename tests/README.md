# Tests

93 tests covering config validation, operator logic, metrics serialization, backend retry, data loading, and the server health endpoint.

```bash
uv run pytest tests/ -v
```

| File | Coverage |
|---|---|
| `test_config.py` | Config schema, model validation, scoring policy, YAML/Python loading |
| `test_operators.py` | All operator registries, routing with N models, selection, recombination prompts, evaluation |
| `test_metrics.py` | Token/routing metrics, flat dict serialization, eval aggregation |
| `test_backend.py` | Retry policy, batched generation |
| `test_data.py` | Parquet/JSON loading, dataset listing |
| `test_server.py` | Health endpoint |
