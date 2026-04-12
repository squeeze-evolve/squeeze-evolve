# Benchmarks

Ready-to-run benchmark configurations reproducing the paper's evaluation. Each subdirectory is self-contained with a `configs/` folder and a `run.sh` script.

## Benchmarks

| Benchmark | Problems | Task | Evaluation | Answer format | Multimodal |
|---|---|---|---|---|---|
| `aime25/` | 30 | math | boxed_math | Integer in `\boxed{}` | No |
| `hmmt25/` | 30 | math | boxed_math | Numerical/symbolic in `\boxed{}` | No |
| `gpqa_diamond/` | 198 | gpqa_diamond | boxed_math | Multiple choice letter in `\boxed{}` | No |
| `babyvision/` | varies | babyvision | babyvision-judge | Answer in `\boxed{}` | Yes (single image) |
| `mmmu_pro/` | varies | mmmu_pro | mmmu_pro-judge | Option letter in `\boxed{}` | Yes (up to 7 images) |

## Structure

Each benchmark directory contains:

```
benchmarks/<name>/
  configs/
    example.yaml      # Annotated template — copy and fill in model endpoints
  register.py          # Operator registration (recombination + evaluation)
  run.sh               # Runs the example config (text-only benchmarks)
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
2. Create `benchmarks/<name>/register.py` — register recombination and evaluation operators.
3. Place the dataset in `data/<name>/`.

## Prerequisites

Each config assumes models are served at the specified `base_url` endpoints. You must start vLLM or another OpenAI-compatible server for each model before running. Update `base_url` and `api_key` in the YAML configs to match your deployment.

---

## Multimodal Benchmarks

### Overview

Multimodal benchmarks (BabyVision, MMMU Pro) extend Squeeze-Evolve to handle vision-language tasks. They use a `MultimodalPrompt` type that carries both text and base64-encoded images through the pipeline.

### How it works

1. **`MultimodalPrompt` type** (`src/squeeze_evolve/core/types.py`): A dataclass with `text: str` and `images: list[str]` (base64 data URLs). All existing text-only paths are unaffected — they continue to use plain strings.

2. **Image handling**: Images are stored as base64 data URLs (`data:image/...;base64,...`) at **original resolution** (no resize). This matches the OpenAI API `image_url` format and keeps parquet files self-contained.

3. **Recombination**: The aggregate prompt is built from `prompt.text` only. Images are included in the OpenAI API call by default (`include_images_in_recombination: true`), so the model sees both the recombination text and the original images.

4. **Evaluation**: Both benchmarks use **LLM-as-judge** (GPT-4o via `judge_model`). The judge receives the candidate's extracted answer and ground truth, and returns True/False.

### Limitations

> **Fitness-based routing is NOT supported for multimodal benchmarks.**
>
> Multimodal runs must use a **single model** (no multi-model routing). The `confidence` fitness signal requires prompt logprobs, which are not available for multimodal prompts via the OpenAI API. Use `fitness: diversity` instead.
>
> When `multimodal: true` is set, scoring and routing are skipped — all groups are processed by the single model.

### Preparing data

```bash
# Install HuggingFace datasets if needed
pip install datasets

# Prepare BabyVision
python benchmarks/prepare_data.py --benchmark babyvision

# Prepare MMMU Pro
python benchmarks/prepare_data.py --benchmark mmmu_pro

# Prepare both
python benchmarks/prepare_data.py --benchmark all
```

### Running multimodal benchmarks

```bash
# BabyVision (single-image VQA)
squeeze-evolve-client \
  --config benchmarks/babyvision/configs/example.yaml \
  --input data/babyvision/test.parquet

# MMMU Pro (multi-image multiple-choice)
squeeze-evolve-client \
  --config benchmarks/mmmu_pro/configs/example.yaml \
  --input data/mmmu_pro/test.parquet
```

### Adding a new vision benchmark

1. Create `benchmarks/<name>/register.py` — register recombination (reuse `make_aggregate_prompt`) and evaluation operators (use `eval_*_judge` pattern from `common.py`).
2. Create `benchmarks/<name>/configs/example.yaml` — set `multimodal: true`, `fitness: diversity`, and configure `judge_model`.
3. Update `benchmarks/prepare_data.py` to download and convert the dataset.
4. Add answer extraction logic to `_extract_answer()` in `orchestrator.py` if needed.

### Config keys for multimodal

| Key | Location | Default | Description |
|---|---|---|---|
| `multimodal` | `routing` | `false` | Enable multimodal prompt pipeline |
| `include_images_in_recombination` | `routing` | `true` | Include images in recombination API calls |
| `judge_model` | top-level | `null` | Model config for LLM-as-judge evaluation |
