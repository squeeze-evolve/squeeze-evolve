"""Configuration schema and validation."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class RetryConfig(BaseModel):
    max_retries: int = 5
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    request_timeout_seconds: float = 120.0
    jitter_seconds: float = 0.5


class RoutingConfig(BaseModel):
    k: int = Field(..., ge=1)
    population: int = Field(..., ge=1)
    groups: Optional[int] = Field(None, ge=1)
    loops: int = Field(..., ge=1)
    confidence_percentiles: list[float] = Field(default_factory=lambda: [50.0])
    fitness: str = "confidence"
    selection: str = "uniform"
    selection_temperature: float = Field(1.0, gt=0.0)
    update: str = "replace"
    lite_fraction: float = Field(0.0, ge=0.0, le=1.0)
    lite_method: str = "majority"
    recombination: str = "aggregate"
    evaluation: str = "none"
    task: str = "math"
    generation_batch_size: int = Field(32, ge=1)
    strip_think: bool = False
    seed: Optional[int] = 0

    # -- Multimodal settings ------------------------------------------------
    multimodal: bool = False
    include_images_in_recombination: bool = True

    @model_validator(mode="after")
    def _default_groups(self) -> RoutingConfig:
        if self.groups is None:
            self.groups = self.population
        return self


class ModelConfig(BaseModel):
    name: str
    base_url: str
    api_key: str = "EMPTY"
    served_model_name: Optional[str] = None
    tokenizer: Optional[str] = None  # HF repo ID; defaults to name
    endpoint: Literal["chat", "completions"] = "chat"
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] = None
    seed: Optional[int] = None
    max_concurrency: int = Field(32, ge=1)
    prompt_logprobs: int = Field(0, ge=0)
    vllm_extensions: bool = False

    @model_validator(mode="after")
    def _validate_endpoint(self) -> ModelConfig:
        if self.endpoint == "completions":
            if self.tokenizer is None:
                self.tokenizer = self.name
        return self


class RunConfig(BaseModel):
    run_name: str = "default"
    routing: RoutingConfig
    models: list[ModelConfig]
    scoring_model: Optional[ModelConfig] = None
    judge_model: Optional[ModelConfig] = None
    retry: RetryConfig = RetryConfig()
    resume: bool = False
    checkpoint_dir: str = "./artifacts/checkpoints"
    metrics_path: str = "./artifacts/metrics.json"

    @model_validator(mode="after")
    def _validate_models(self) -> RunConfig:
        if len(self.models) < 1:
            raise ValueError("At least one model required")
        return self

    @property
    def model_count(self) -> int:
        return len(self.models)


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------

def validate_scoring_policy(cfg: RunConfig) -> None:
    """Enforce scoring-model constraints based on fitness mode and model count."""
    # Multimodal benchmarks skip fitness-based routing entirely.
    if cfg.routing.multimodal:
        return

    if cfg.routing.fitness == "diversity":
        return

    is_multi = cfg.model_count > 1
    scoring = cfg.scoring_model

    if is_multi:
        if scoring is None:
            raise ValueError("Multi-model runs require explicit `scoring_model`.")
        if not scoring.vllm_extensions:
            raise ValueError("Cross-model confidence scoring requires a scoring model with vllm_extensions enabled.")
        return

    if cfg.models[-1].prompt_logprobs > 0:
        return

    if scoring is None or not scoring.vllm_extensions:
        raise ValueError(
            "Single-model runs without prompt_logprobs require a scoring_model "
            "with vllm_extensions because the target cannot provide logprobs."
        )
