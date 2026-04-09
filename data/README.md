# Datasets

Evaluation datasets in parquet format. Each file must contain at minimum an `orig_prompt` column. The `gt` column is optional and only required when `evaluation` is set to something other than `none`.

| Dataset | Problems | Task type |
|---|---|---|
| `aime25/test.parquet` | 30 | math |
| `hmmt25/test.parquet` | 30 | math |
| `gpqa_diamond/test.parquet` | 198 | gpqa_diamond |

## Supported formats

The `load_dataset()` function accepts `.parquet`, `.jsonl`, and `.json` files. Use `--n-problems` on the CLI to limit the number of problems loaded.
