#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

squeeze-evolve-client \
    --config "$SCRIPT_DIR/configs/example.yaml" \
    --input "$REPO_ROOT/data/aime25/test.parquet" \
    --output "$REPO_ROOT/results/aime25/example.json"
