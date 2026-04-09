# Benchmarks

Ready-to-run benchmark configurations reproducing the paper's evaluation. Each subdirectory is self-contained with a `configs/` folder and a `run.sh` script.

## Benchmarks

| Benchmark | Problems | Task | Evaluation | Answer format |
|---|---|---|---|---|
| `aime25/` | 30 | math | boxed_math | Integer in `\boxed{}` |
| `hmmt25/` | 30 | math | boxed_math | Numerical/symbolic in `\boxed{}` |
| `gpqa_diamond/` | 198 | gpqa_diamond | boxed_math | Multiple choice letter in `\boxed{}` |

## Structure

Each benchmark directory contains:

```
benchmarks/<name>/
  configs/
    example.yaml      # Annotated template — copy and fill in model endpoints
  run.sh               # Runs the example config
```

## Shared settings (from the paper)

All benchmarks use the same evolutionary parameters unless noted:

| Parameter | Value |
|---|---|
| Population (N) | 16 |
| Group size (K) | 4 |
| Loops (T) | 10 |
| Seeds | 4 (averaged) |
| Selection | Uniform |
| Update | Replace |
| Fitness | Group Confidence (GC) |

## Running a benchmark

```bash
# Single config
squeeze-evolve-client \
  --config benchmarks/aime25/configs/example.yaml \
  --input data/aime25/test.parquet \
  --output results/aime25/example.json

# All configs for a benchmark (4 seeds each)
bash benchmarks/aime25/run.sh
bash benchmarks/hmmt25/run.sh
bash benchmarks/gpqa_diamond/run.sh
```

## Adding a new benchmark

1. Create `benchmarks/<name>/configs/example.yaml` — copy from a similar benchmark and update `task`, `evaluation`, and paths.
2. Create `benchmarks/<name>/run.sh` — copy from a similar benchmark and update `DATA` and `RESULTS_DIR`.
3. Place the dataset in `data/<name>/`.

## Prerequisites

Each config assumes models are served at the specified `base_url` endpoints. You must start vLLM or another OpenAI-compatible server for each model before running. Update `base_url` and `api_key` in the YAML configs to match your deployment.
