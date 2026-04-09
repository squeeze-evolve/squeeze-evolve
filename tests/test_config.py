import os
import tempfile

import pytest

from squeeze_evolve.core.config import (
    ModelConfig,
    RoutingConfig,
    RunConfig,
    validate_scoring_policy,
)
from squeeze_evolve.algorithm.operators import configs
from squeeze_evolve.algorithm.utils import load_run_config


def _model(prompt_logprobs: bool = False, vllm_extensions: bool = False) -> ModelConfig:
    return ModelConfig(
        name="test-model",
        base_url="http://localhost:8000/v1",
        prompt_logprobs=prompt_logprobs,
        vllm_extensions=vllm_extensions,
    )


# --- schema ---

def test_defaults() -> None:
    model = ModelConfig(name="test", base_url="http://localhost:8000/v1")
    assert model.reasoning_effort is None
    assert model.prompt_logprobs is False
    assert model.vllm_extensions is False


def test_reasoning_effort_flag() -> None:
    model = ModelConfig(name="test", base_url="http://localhost:8000/v1", reasoning_effort="medium")
    assert model.reasoning_effort == "medium"


def test_logprobs_flag() -> None:
    model = ModelConfig(name="test", base_url="http://localhost:8000/v1", prompt_logprobs=True)
    assert model.prompt_logprobs is True


def test_groups_defaults_to_population() -> None:
    rc = RoutingConfig(k=2, population=8, loops=3)
    assert rc.groups == 8


def test_groups_explicit() -> None:
    rc = RoutingConfig(k=2, population=8, groups=4, loops=3)
    assert rc.groups == 4


def test_model_count_single() -> None:
    cfg = RunConfig(
        routing=RoutingConfig(k=1, population=2, loops=2),
        models=[_model(prompt_logprobs=True)],
    )
    assert cfg.model_count == 1


def test_model_count_two() -> None:
    cfg = RunConfig(
        routing=RoutingConfig(k=1, population=2, loops=2),
        models=[_model(), _model()],
    )
    assert cfg.model_count == 2


def test_model_count_three() -> None:
    cfg = RunConfig(
        routing=RoutingConfig(k=1, population=2, loops=2, confidence_percentiles=[30.0, 60.0]),
        models=[_model(), _model(), _model()],
    )
    assert cfg.model_count == 3


def test_empty_models_raises() -> None:
    with pytest.raises(ValueError, match="At least one model"):
        RunConfig(
            routing=RoutingConfig(k=1, population=2, loops=2),
            models=[],
        )


# --- policy ---

def test_single_with_logprobs_requires_no_scoring_model() -> None:
    cfg = RunConfig(
        routing=RoutingConfig(k=1, population=2, loops=2),
        models=[_model(prompt_logprobs=True)],
    )
    validate_scoring_policy(cfg)


def test_multi_model_requires_vllm_scorer() -> None:
    cfg = RunConfig(
        routing=RoutingConfig(k=1, population=2, loops=2),
        models=[_model(prompt_logprobs=True), _model(prompt_logprobs=True)],
    )
    with pytest.raises(ValueError):
        validate_scoring_policy(cfg)


def test_single_without_logprobs_requires_vllm_scorer() -> None:
    cfg = RunConfig(
        routing=RoutingConfig(k=1, population=2, loops=2),
        models=[_model()],
    )
    with pytest.raises(ValueError):
        validate_scoring_policy(cfg)


def test_generation_batch_size_default() -> None:
    rc = RoutingConfig(k=1, population=2, loops=1)
    assert rc.generation_batch_size == 32


def test_max_concurrency_default() -> None:
    model = ModelConfig(name="test", base_url="http://localhost:8000/v1")
    assert model.max_concurrency == 32


def test_diversity_fitness_skips_scoring_validation() -> None:
    cfg = RunConfig(
        routing=RoutingConfig(k=1, population=2, loops=2, fitness="diversity"),
        models=[_model()],
    )
    validate_scoring_policy(cfg)


def test_confidence_percentiles_default() -> None:
    rc = RoutingConfig(k=1, population=2, loops=1)
    assert rc.confidence_percentiles == [50.0]


# --- config loading: yaml ---

def testvalidate_problems_missing_orig_prompt() -> None:
    from squeeze_evolve.algorithm.utils import validate_problems
    with pytest.raises(ValueError, match="Problem 1 missing"):
        validate_problems([{"orig_prompt": "ok"}, {"gt": "no prompt"}])


def testvalidate_problems_valid() -> None:
    from squeeze_evolve.algorithm.utils import validate_problems
    validate_problems([{"orig_prompt": "q1"}, {"orig_prompt": "q2", "gt": "a2"}])


def test_load_unknown_name_raises() -> None:
    with pytest.raises((KeyError, FileNotFoundError)):
        load_run_config("nonexistent_preset")


def test_load_yaml_hetero() -> None:
    cfg = load_run_config("benchmarks/aime25/configs/example.yaml")
    assert cfg.model_count == 2
    assert cfg.models[0].name == "<cheap_model>"
    assert cfg.models[-1].name == "<expensive_model>"


def test_load_yaml_gpqa() -> None:
    cfg = load_run_config("benchmarks/gpqa_diamond/configs/example.yaml")
    assert cfg.routing.task == "gpqa_diamond"
    assert cfg.model_count == 2


# --- config loading: .py file ---

def test_load_yaml_file() -> None:
    cfg = load_run_config("benchmarks/hmmt25/configs/example.yaml")
    assert isinstance(cfg, RunConfig)
    assert cfg.routing.task == "math"


def test_load_py_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(
            "from squeeze_evolve.core.config import RunConfig, ModelConfig, RoutingConfig\n"
            "config = RunConfig(\n"
            "    routing=RoutingConfig(k=1, population=2, loops=1),\n"
            "    models=[ModelConfig(name='t', base_url='http://localhost:8000/v1', prompt_logprobs=True)],\n"
            ")\n"
        )
        f.flush()
        cfg = load_run_config(f.name)
    os.unlink(f.name)
    assert isinstance(cfg, RunConfig)


# --- config loading: include_path ---

def test_include_path_discovers_configs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, "my_preset.py")
        with open(py_file, "w") as f:
            f.write(
                "from squeeze_evolve import configs\n"
                "from squeeze_evolve.core.config import RunConfig, ModelConfig, RoutingConfig\n"
                "@configs.register('_test_discovered')\n"
                "def _test_discovered():\n"
                "    return RunConfig(\n"
                "        routing=RoutingConfig(k=1, population=2, loops=1),\n"
                "        models=[ModelConfig(name='t', base_url='http://localhost:8000/v1', prompt_logprobs=True)],\n"
                "    )\n"
            )
        cfg = load_run_config("_test_discovered", include_path=tmpdir)
        assert cfg.routing.k == 1


def test_endpoint_defaults_to_chat() -> None:
    model = ModelConfig(name="test", base_url="http://localhost:8000/v1")
    assert model.endpoint == "chat"


def test_endpoint_completions_accepted() -> None:
    model = ModelConfig(name="test", base_url="http://localhost:8000/v1", endpoint="completions")
    assert model.endpoint == "completions"
    assert model.tokenizer == "test"  # auto-defaulted to name


def test_endpoint_completions_with_reasoning_effort() -> None:
    model = ModelConfig(
        name="test",
        base_url="http://localhost:8000/v1",
        endpoint="completions",
        reasoning_effort="medium",
    )
    assert model.reasoning_effort == "medium"
    assert model.endpoint == "completions"


def test_invalid_endpoint_value() -> None:
    with pytest.raises(Exception):
        ModelConfig(name="test", base_url="http://localhost:8000/v1", endpoint="invalid")


def test_load_yaml_two_model() -> None:
    cfg = load_run_config("benchmarks/aime25/configs/example.yaml")
    assert cfg.model_count == 2
    assert cfg.routing.confidence_percentiles == [10.0]
    assert cfg.models[0].name == "<cheap_model>"
    assert cfg.models[1].name == "<expensive_model>"
