from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import uuid

MODEL_PRICING = {
    # Anthropic
    "anthropic/claude-haiku-4.5":           {"input": 0.80,  "output": 4.00},
    "anthropic/claude-sonnet-4.6":          {"input": 3.00,  "output": 15.00},
    "anthropic/claude-opus-4.6":            {"input": 15.00, "output": 75.00},
    "anthropic/claude-3.7-sonnet:thinking": {"input": 3.00,  "output": 15.00},
    # OpenAI
    "openai/gpt-4o-mini":    {"input": 0.15,  "output": 0.60},
    "openai/gpt-4o":         {"input": 2.50,  "output": 10.00},
    "openai/gpt-4.1":        {"input": 2.00,  "output": 8.00},
    "openai/gpt-4.1-mini":   {"input": 0.40,  "output": 1.60},
    "openai/o3-mini":        {"input": 1.10,  "output": 4.40},
    "openai/o3":             {"input": 10.00, "output": 40.00},
    "openai/o4-mini":        {"input": 1.10,  "output": 4.40},
    # Google
    "google/gemini-2.5-flash":      {"input": 0.15, "output": 0.60},
    "google/gemini-2.5-flash-lite": {"input": 0.075,"output": 0.30},
    "google/gemini-2.5-pro":        {"input": 1.25, "output": 10.00},
    # DeepSeek
    "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek/deepseek-r1":   {"input": 0.55, "output": 2.19},
    "deepseek/deepseek-v3.2": {"input": 0.27, "output": 1.10},
    # Moonshot
    "moonshotai/kimi-k2":         {"input": 0.14, "output": 0.14},
    "moonshotai/kimi-k2.5":       {"input": 0.14, "output": 0.14},
    "moonshotai/kimi-k2-thinking": {"input": 0.14, "output": 0.14},
    # Meta
    "meta-llama/llama-4-maverick":        {"input": 0.22, "output": 0.88},
    "meta-llama/llama-4-scout":           {"input": 0.11, "output": 0.34},
    "meta-llama/llama-3.3-70b-instruct":  {"input": 0.40, "output": 0.40},
    # Mistral
    "mistralai/mistral-large":                  {"input": 2.00, "output": 6.00},
    "mistralai/mistral-medium-3":               {"input": 0.40, "output": 2.00},
    "mistralai/mistral-small-3.2-24b-instruct": {"input": 0.10, "output": 0.30},
    # xAI
    "x-ai/grok-3":      {"input": 3.00,  "output": 15.00},
    "x-ai/grok-3-mini": {"input": 0.30,  "output": 0.50},
    "x-ai/grok-4":      {"input": 3.00,  "output": 15.00},
    # Qwen
    "qwen/qwen3-235b-a22b":  {"input": 0.14, "output": 0.60},
    "qwen/qwen3-30b-a3b":    {"input": 0.04, "output": 0.10},
    "qwen/qwq-32b":          {"input": 0.15, "output": 0.60},
    # Amazon
    "amazon/nova-pro-v1":  {"input": 0.80, "output": 3.20},
    "amazon/nova-lite-v1": {"input": 0.06, "output": 0.24},
    # Cohere
    "cohere/command-a": {"input": 2.50, "output": 10.00},
    # Perplexity
    "perplexity/sonar-pro": {"input": 3.00, "output": 15.00},
}

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING[model]
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


class ExperimentResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: Optional[str] = None
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


class Variant(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # 'system_prompt' | 'prompt_length'
    name: str
    content: str
    paused: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class ExperimentConfig(BaseModel):
    name: str
    model: str = DEFAULT_MODEL
    prompt: str
    system_prompt: str = ""
    num_runs: int = 3
    temperature: Optional[float] = None
    max_tokens: int = 1024
