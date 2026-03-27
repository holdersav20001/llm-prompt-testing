from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import uuid

MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}

DEFAULT_MODEL = "claude-sonnet-4-6"


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING[model]
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


class ExperimentResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment: str
    timestamp: datetime = Field(default_factory=datetime.now)
    model: str
    prompt: str
    system_prompt: str = ""
    response_text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    ttft_ms: Optional[int] = None
    temperature: Optional[float] = None
    params: dict = Field(default_factory=dict)
    run_index: int = 1


class ExperimentConfig(BaseModel):
    name: str
    model: str = DEFAULT_MODEL
    prompt: str
    system_prompt: str = ""
    num_runs: int = 3
    temperature: Optional[float] = None
    max_tokens: int = 1024
