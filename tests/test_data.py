import json
import tempfile

from squeeze_evolve.core.data import list_datasets, load_dataset, load_parquet


def test_load_json_problems() -> None:
    data = [{"orig_prompt": "What is 2+2?", "gt": "4"}, {"orig_prompt": "Solve x=1", "gt": "1"}]
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(data, f)
        f.flush()
        problems = load_dataset(f.name)
    assert len(problems) == 2
    assert "orig_prompt" in problems[0]


def test_load_parquet_aime25() -> None:
    problems = load_parquet("data/aime25/test.parquet")
    assert len(problems) == 30
    assert "orig_prompt" in problems[0]
    assert isinstance(problems[0]["orig_prompt"], str)
    assert len(problems[0]["orig_prompt"]) > 10


def test_load_parquet_gpqa() -> None:
    problems = load_parquet("data/gpqa_diamond/test.parquet", n_problems=5)
    assert len(problems) == 5
    assert problems[0]["orig_prompt"]


def test_load_parquet_rg() -> None:
    problems = load_parquet("data/rg_cognition/test.parquet", n_problems=3)
    assert len(problems) == 3


def test_load_parquet_supergpqa() -> None:
    problems = load_parquet("data/supergpqa/supergpqa_test_1.parquet", n_problems=3)
    assert len(problems) == 3


def test_load_dataset_auto_parquet() -> None:
    problems = load_dataset("data/hmmt25/test.parquet", n_problems=2)
    assert len(problems) == 2
    assert "orig_prompt" in problems[0]


def test_load_dataset_n_problems_limit() -> None:
    problems = load_dataset("data/aime25/test.parquet", n_problems=5)
    assert len(problems) == 5


def test_gt_extracted() -> None:
    problems = load_parquet("data/aime25/test.parquet", n_problems=1)
    assert problems[0]["gt"] is not None


def test_list_datasets() -> None:
    datasets = list_datasets("data")
    assert len(datasets) > 0
    assert any("aime25" in d for d in datasets)
    assert any("gpqa_diamond" in d for d in datasets)
