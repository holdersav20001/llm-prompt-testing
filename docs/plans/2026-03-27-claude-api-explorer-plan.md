# Claude API Explorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Streamlit app that runs 5 Claude API experiments, stores results in SQLite, visualises with Plotly, and generates copyable blog post content.

**Architecture:** Content-led design: 5 experiments (temperature sweep, system prompts, model comparison, token economics, streaming) each mapping to one blog post. Streamlit UI with 3 tabs (Run, History, Post Content). All Claude API calls go through a single `ClaudeClient` wrapper with retry and cost tracking.

**Tech Stack:** Python 3.11+, anthropic SDK, streamlit, plotly, pandas, pydantic v2, python-dotenv, sqlite3 (stdlib), pytest, pytest-mock

---

## Phase 1: Foundation

### Task 1: Project skeleton

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/experiments/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/.gitkeep`

**Step 1: Create requirements.txt**

```
anthropic>=0.40.0
streamlit>=1.40.0
plotly>=5.24.0
pandas>=2.2.0
pydantic>=2.9.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-mock>=3.14.0
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "claude-api-explorer"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 3: Create .env.example**

```
ANTHROPIC_API_KEY=your-key-here
```

**Step 4: Create .gitignore**

```
venv/
.env
data/results.db
__pycache__/
*.pyc
.pytest_cache/
```

**Step 5: Create empty __init__.py files**

Create `src/__init__.py`, `src/experiments/__init__.py`, `tests/__init__.py` — all empty.

Create `data/.gitkeep` — empty.

**Step 6: Set up virtual environment and install dependencies**

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

Expected: all packages install without errors.

**Step 7: Verify pytest works**

```bash
pytest
```

Expected: `no tests ran` (0 errors, 0 failures).

**Step 8: Commit**

```bash
git init
git add requirements.txt pyproject.toml .env.example .gitignore src/ tests/ data/.gitkeep docs/
git commit -m "chore: project skeleton"
```

---

### Task 2: config.py — Pydantic models and pricing constants

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
from src.config import ExperimentResult, ExperimentConfig, MODEL_PRICING, calculate_cost

def test_experiment_result_defaults():
    result = ExperimentResult(
        experiment="test",
        model="claude-sonnet-4-6",
        prompt="hello",
        response_text="world",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.001,
        latency_ms=500,
    )
    assert result.id != ""
    assert result.timestamp is not None
    assert result.run_index == 1
    assert result.system_prompt == ""
    assert result.ttft_ms is None

def test_calculate_cost_haiku():
    cost = calculate_cost("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(cost - 4.80) < 0.001  # 0.80 + 4.00

def test_calculate_cost_sonnet():
    cost = calculate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(cost - 18.00) < 0.001  # 3.00 + 15.00

def test_experiment_config_defaults():
    config = ExperimentConfig(name="my_test", prompt="what is 2+2?")
    assert config.model == "claude-sonnet-4-6"
    assert config.num_runs == 3
    assert config.max_tokens == 1024
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `ImportError` — `src.config` does not exist yet.

**Step 3: Write implementation**

```python
# src/config.py
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: 4 tests PASS.

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add Pydantic config models and pricing constants"
```

---

## Phase 2: Core Infrastructure

### Task 3: storage.py — SQLite storage

**Files:**
- Create: `src/storage.py`
- Create: `tests/test_storage.py`

**Step 1: Write the failing tests**

```python
# tests/test_storage.py
import pytest
from src.storage import SQLiteStorage
from src.config import ExperimentResult

@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "test.db")
    return SQLiteStorage(db_path=db_path)

@pytest.fixture
def sample_result():
    return ExperimentResult(
        experiment="temperature_sweep",
        model="claude-sonnet-4-6",
        prompt="hello",
        response_text="world",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.001,
        latency_ms=500,
        temperature=0.5,
    )

def test_save_and_retrieve(storage, sample_result):
    storage.save(sample_result)
    results = storage.get_by_experiment("temperature_sweep")
    assert len(results) == 1
    assert results[0].id == sample_result.id
    assert results[0].response_text == "world"

def test_get_all_empty(storage):
    assert storage.get_all() == []

def test_get_by_experiment_filters(storage, sample_result):
    storage.save(sample_result)
    other = ExperimentResult(
        experiment="streaming",
        model="claude-sonnet-4-6",
        prompt="hi",
        response_text="hello",
        input_tokens=5,
        output_tokens=5,
        cost_usd=0.0001,
        latency_ms=100,
    )
    storage.save(other)
    assert len(storage.get_by_experiment("temperature_sweep")) == 1
    assert len(storage.get_by_experiment("streaming")) == 1
    assert len(storage.get_all()) == 2

def test_temperature_is_queryable_column(storage, sample_result):
    storage.save(sample_result)
    import sqlite3
    with sqlite3.connect(storage.db_path) as conn:
        row = conn.execute("SELECT temperature FROM results WHERE id = ?", (sample_result.id,)).fetchone()
    assert row[0] == 0.5
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_storage.py -v
```

Expected: `ImportError`.

**Step 3: Write implementation**

```python
# src/storage.py
import sqlite3
import json
from pathlib import Path
from src.config import ExperimentResult


class SQLiteStorage:
    def __init__(self, db_path: str = "data/results.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    experiment TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL,
                    cost_usd REAL NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    result_json TEXT NOT NULL
                )
            """)

    def save(self, result: ExperimentResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.id,
                    result.experiment,
                    result.timestamp.isoformat(),
                    result.model,
                    result.temperature,
                    result.cost_usd,
                    result.latency_ms,
                    result.model_dump_json(),
                ),
            )

    def get_by_experiment(self, experiment: str) -> list[ExperimentResult]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT result_json FROM results WHERE experiment = ? ORDER BY timestamp",
                (experiment,),
            ).fetchall()
        return [ExperimentResult.model_validate_json(row[0]) for row in rows]

    def get_all(self) -> list[ExperimentResult]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT result_json FROM results ORDER BY timestamp DESC"
            ).fetchall()
        return [ExperimentResult.model_validate_json(row[0]) for row in rows]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_storage.py -v
```

Expected: 4 tests PASS.

**Step 5: Commit**

```bash
git add src/storage.py tests/test_storage.py
git commit -m "feat: add SQLite storage"
```

---

### Task 4: client.py — Claude API wrapper

**Files:**
- Create: `src/client.py`
- Create: `tests/test_client.py`

**Step 1: Write the failing tests**

```python
# tests/test_client.py
import pytest
from unittest.mock import MagicMock, patch, call
from anthropic import APIStatusError
from src.client import ClaudeClient
from src.config import calculate_cost

@pytest.fixture
def client():
    return ClaudeClient(api_key="test-key")

def make_mock_response(input_tokens=100, output_tokens=50, text="hello"):
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response

def test_call_returns_result(client, mocker):
    mock_create = mocker.patch.object(
        client._client.messages, "create",
        return_value=make_mock_response(100, 50, "hello world"),
    )
    result = client.call(model="claude-sonnet-4-6", prompt="hi", max_tokens=100)
    assert result["response_text"] == "hello world"
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["cost_usd"] > 0
    assert result["latency_ms"] >= 0

def test_cost_calculation_correct(client):
    cost = calculate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
    expected = (1000 * 3.00 + 500 * 15.00) / 1_000_000
    assert abs(cost - expected) < 1e-9

def test_retry_on_rate_limit(client, mocker):
    mock_status_error = APIStatusError(
        "rate limit", response=MagicMock(status_code=429), body={}
    )
    mock_create = mocker.patch.object(
        client._client.messages, "create",
        side_effect=[mock_status_error, mock_status_error, make_mock_response()],
    )
    mocker.patch("time.sleep")
    result = client.call(model="claude-sonnet-4-6", prompt="hi", max_tokens=100)
    assert result["response_text"] is not None
    assert mock_create.call_count == 3

def test_auth_error_no_retry(client, mocker):
    mock_status_error = APIStatusError(
        "auth", response=MagicMock(status_code=401), body={}
    )
    mocker.patch.object(client._client.messages, "create", side_effect=mock_status_error)
    mocker.patch("time.sleep")
    with pytest.raises(APIStatusError):
        client.call(model="claude-sonnet-4-6", prompt="hi", max_tokens=100)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_client.py -v
```

Expected: `ImportError`.

**Step 3: Write implementation**

```python
# src/client.py
import time
import anthropic
from anthropic import APIStatusError
from src.config import MODEL_PRICING, calculate_cost


class ClaudeClient:
    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
        **kwargs,
    ) -> dict:
        messages = [{"role": "user", "content": prompt}]
        api_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if system_prompt:
            api_kwargs["system"] = system_prompt

        for attempt in range(3):
            try:
                start = time.time()
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    **api_kwargs,
                )
                latency_ms = int((time.time() - start) * 1000)
                return {
                    "response_text": response.content[0].text,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cost_usd": calculate_cost(model, response.usage.input_tokens, response.usage.output_tokens),
                    "latency_ms": latency_ms,
                }
            except APIStatusError as e:
                if e.status_code == 401:
                    raise
                if e.status_code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    def stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        messages = [{"role": "user", "content": prompt}]
        api_kwargs = {}
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if system_prompt:
            api_kwargs["system"] = system_prompt

        ttft_ms = None
        collected = []
        start = time.time()

        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            **api_kwargs,
        ) as stream:
            for text in stream.text_stream:
                if ttft_ms is None:
                    ttft_ms = int((time.time() - start) * 1000)
                collected.append(text)
            final = stream.get_final_message()

        latency_ms = int((time.time() - start) * 1000)
        return {
            "response_text": "".join(collected),
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
            "cost_usd": calculate_cost(model, final.usage.input_tokens, final.usage.output_tokens),
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_client.py -v
```

Expected: 4 tests PASS.

**Step 5: Commit**

```bash
git add src/client.py tests/test_client.py
git commit -m "feat: add Claude API client with retry and cost tracking"
```

---

### Task 5: experiments/base.py — BaseExperiment

**Files:**
- Create: `src/experiments/base.py`
- Modify: `tests/test_experiments.py` (create if not exists)

**Step 1: Write the failing test**

```python
# tests/test_experiments.py
import pytest
from unittest.mock import MagicMock
from src.experiments.base import BaseExperiment
from src.config import ExperimentConfig, ExperimentResult
from src.storage import SQLiteStorage

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.call.return_value = {
        "response_text": "test response",
        "input_tokens": 50,
        "output_tokens": 30,
        "cost_usd": 0.001,
        "latency_ms": 200,
    }
    return client

@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(db_path=str(tmp_path / "test.db"))

@pytest.fixture
def config():
    return ExperimentConfig(name="test_exp", prompt="test prompt", temperature=0.5)

def test_run_single_returns_result(mock_client, storage, config):
    exp = BaseExperiment(mock_client, storage, config)
    result = exp.run_single(run_index=1)
    assert isinstance(result, ExperimentResult)
    assert result.experiment == "test_exp"
    assert result.run_index == 1
    assert result.response_text == "test response"

def test_run_single_saves_to_storage(mock_client, storage, config):
    exp = BaseExperiment(mock_client, storage, config)
    result = exp.run_single(run_index=1)
    saved = storage.get_by_experiment("test_exp")
    assert len(saved) == 1
    assert saved[0].id == result.id

def test_run_single_override_model(mock_client, storage, config):
    exp = BaseExperiment(mock_client, storage, config)
    exp.run_single(run_index=1, model="claude-haiku-4-5-20251001")
    call_args = mock_client.call.call_args
    assert call_args.kwargs["model"] == "claude-haiku-4-5-20251001"

def test_run_raises_not_implemented(mock_client, storage, config):
    exp = BaseExperiment(mock_client, storage, config)
    with pytest.raises(NotImplementedError):
        exp.run()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_experiments.py -v
```

Expected: `ImportError`.

**Step 3: Write implementation**

```python
# src/experiments/base.py
from src.client import ClaudeClient
from src.storage import SQLiteStorage
from src.config import ExperimentConfig, ExperimentResult


class BaseExperiment:
    def __init__(self, client: ClaudeClient, storage: SQLiteStorage, config: ExperimentConfig):
        self.client = client
        self.storage = storage
        self.config = config

    def run(self) -> list[ExperimentResult]:
        raise NotImplementedError

    def run_single(self, run_index: int = 1, extra_params: dict = None, **overrides) -> ExperimentResult:
        call_kwargs = {
            "model": self.config.model,
            "prompt": self.config.prompt,
            "system_prompt": self.config.system_prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        call_kwargs.update(overrides)

        raw = self.client.call(**call_kwargs)

        result = ExperimentResult(
            experiment=self.config.name,
            model=call_kwargs["model"],
            prompt=call_kwargs["prompt"],
            system_prompt=call_kwargs.get("system_prompt", ""),
            temperature=call_kwargs.get("temperature"),
            run_index=run_index,
            params=extra_params or {},
            **raw,
        )
        self.storage.save(result)
        return result
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_experiments.py -v
```

Expected: 4 tests PASS.

**Step 5: Commit**

```bash
git add src/experiments/base.py tests/test_experiments.py
git commit -m "feat: add BaseExperiment class"
```

---

## Phase 3: Experiments

### Task 6: experiments/temperature.py

**Files:**
- Create: `src/experiments/temperature.py`
- Modify: `tests/test_experiments.py`

**Step 1: Add failing tests to tests/test_experiments.py**

```python
# append to tests/test_experiments.py
from src.experiments.temperature import TemperatureSweepExperiment

def test_temperature_sweep_runs_correct_count(mock_client, storage):
    config = ExperimentConfig(name="temperature_sweep", prompt="test", num_runs=2)
    exp = TemperatureSweepExperiment(mock_client, storage, config)
    results = exp.run()
    # 6 temperatures × 2 runs = 12
    assert len(results) == 12

def test_temperature_sweep_covers_all_temps(mock_client, storage):
    config = ExperimentConfig(name="temperature_sweep", prompt="test", num_runs=1)
    exp = TemperatureSweepExperiment(mock_client, storage, config)
    results = exp.run()
    temps = sorted({r.temperature for r in results})
    assert temps == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def test_similarity_identical_texts():
    sim = TemperatureSweepExperiment.similarity(["hello world", "hello world"])
    assert sim == 1.0

def test_similarity_different_texts():
    sim = TemperatureSweepExperiment.similarity(["hello", "completely different text here"])
    assert sim < 0.5
```

**Step 2: Run new tests to verify they fail**

```bash
pytest tests/test_experiments.py::test_temperature_sweep_runs_correct_count -v
```

Expected: `ImportError`.

**Step 3: Write implementation**

```python
# src/experiments/temperature.py
from difflib import SequenceMatcher
from src.experiments.base import BaseExperiment
from src.config import ExperimentConfig, ExperimentResult


class TemperatureSweepExperiment(BaseExperiment):
    TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def run(self) -> list[ExperimentResult]:
        results = []
        for temp in self.TEMPERATURES:
            for i in range(self.config.num_runs):
                result = self.run_single(run_index=i + 1, temperature=temp)
                results.append(result)
        return results

    @staticmethod
    def similarity(texts: list[str]) -> float:
        if len(texts) < 2:
            return 1.0
        scores = [
            SequenceMatcher(None, texts[i], texts[j]).ratio()
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        ]
        return sum(scores) / len(scores)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_experiments.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add src/experiments/temperature.py tests/test_experiments.py
git commit -m "feat: add temperature sweep experiment"
```

---

### Task 7: experiments/system_prompts.py

**Files:**
- Create: `src/experiments/system_prompts.py`
- Modify: `tests/test_experiments.py`

**Step 1: Add failing tests**

```python
# append to tests/test_experiments.py
from src.experiments.system_prompts import SystemPromptsExperiment, SYSTEM_PROMPT_VARIANTS

def test_system_prompts_runs_all_variants(mock_client, storage):
    config = ExperimentConfig(name="system_prompts", prompt="test")
    exp = SystemPromptsExperiment(mock_client, storage, config)
    results = exp.run()
    assert len(results) == len(SYSTEM_PROMPT_VARIANTS)

def test_system_prompts_tags_variant(mock_client, storage):
    config = ExperimentConfig(name="system_prompts", prompt="test")
    exp = SystemPromptsExperiment(mock_client, storage, config)
    results = exp.run()
    variants = [r.params["variant"] for r in results]
    assert "none" in variants
    assert "persona_constraints" in variants
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_experiments.py::test_system_prompts_runs_all_variants -v
```

**Step 3: Write implementation**

```python
# src/experiments/system_prompts.py
from src.experiments.base import BaseExperiment
from src.config import ExperimentResult

SYSTEM_PROMPT_VARIANTS = [
    ("none", ""),
    ("short_role", "You are a helpful assistant."),
    ("detailed_role", "You are an expert technical writer who produces clear, well-structured, accurate responses."),
    ("format_constraints", "You are a helpful assistant. Always respond in bullet points. Keep responses under 100 words."),
    ("persona_constraints", "You are Alex, a senior software engineer with 10 years of experience. Respond in a friendly but direct tone. Structure your response with: 1) Direct answer 2) Brief explanation 3) One concrete example."),
]


class SystemPromptsExperiment(BaseExperiment):
    def run(self) -> list[ExperimentResult]:
        results = []
        for variant_name, system_prompt in SYSTEM_PROMPT_VARIANTS:
            result = self.run_single(
                run_index=1,
                extra_params={"variant": variant_name},
                system_prompt=system_prompt,
                temperature=0.0,
            )
            results.append(result)
        return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_experiments.py -v
```

**Step 5: Commit**

```bash
git add src/experiments/system_prompts.py tests/test_experiments.py
git commit -m "feat: add system prompts experiment"
```

---

### Task 8: experiments/model_compare.py

**Files:**
- Create: `src/experiments/model_compare.py`
- Modify: `tests/test_experiments.py`

**Step 1: Add failing tests**

```python
# append to tests/test_experiments.py
from src.experiments.model_compare import ModelCompareExperiment

def test_model_compare_default_excludes_opus(mock_client, storage):
    config = ExperimentConfig(name="model_compare", prompt="test")
    exp = ModelCompareExperiment(mock_client, storage, config)
    results = exp.run()
    models_used = [r.model for r in results]
    assert "claude-opus-4-6" not in models_used
    assert len(results) == 2

def test_model_compare_includes_opus_when_opt_in(mock_client, storage):
    config = ExperimentConfig(name="model_compare", prompt="test")
    exp = ModelCompareExperiment(mock_client, storage, config, include_opus=True)
    results = exp.run()
    assert len(results) == 3
    assert any(r.model == "claude-opus-4-6" for r in results)
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_experiments.py::test_model_compare_default_excludes_opus -v
```

**Step 3: Write implementation**

```python
# src/experiments/model_compare.py
from src.experiments.base import BaseExperiment
from src.config import ExperimentResult

DEFAULT_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]


class ModelCompareExperiment(BaseExperiment):
    def __init__(self, client, storage, config, include_opus: bool = False):
        super().__init__(client, storage, config)
        self.models = DEFAULT_MODELS + (["claude-opus-4-6"] if include_opus else [])

    def run(self) -> list[ExperimentResult]:
        results = []
        for model in self.models:
            result = self.run_single(run_index=1, model=model, temperature=0.0)
            results.append(result)
        return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_experiments.py -v
```

**Step 5: Commit**

```bash
git add src/experiments/model_compare.py tests/test_experiments.py
git commit -m "feat: add model comparison experiment"
```

---

### Task 9: experiments/token_tracking.py

**Files:**
- Create: `src/experiments/token_tracking.py`
- Modify: `tests/test_experiments.py`

**Step 1: Add failing tests**

```python
# append to tests/test_experiments.py
from src.experiments.token_tracking import TokenTrackingExperiment, PROMPT_VARIANTS

def test_token_tracking_runs_all_combinations(mock_client, storage):
    config = ExperimentConfig(name="token_tracking", prompt="ignored")
    exp = TokenTrackingExperiment(mock_client, storage, config)
    results = exp.run()
    # 3 prompts × 2 models = 6
    assert len(results) == 6

def test_token_tracking_tags_prompt_length(mock_client, storage):
    config = ExperimentConfig(name="token_tracking", prompt="ignored")
    exp = TokenTrackingExperiment(mock_client, storage, config)
    results = exp.run()
    lengths = {r.params["prompt_length"] for r in results}
    assert lengths == {"short", "medium", "long"}
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_experiments.py::test_token_tracking_runs_all_combinations -v
```

**Step 3: Write implementation**

```python
# src/experiments/token_tracking.py
from src.experiments.base import BaseExperiment
from src.config import ExperimentResult

PROMPT_VARIANTS = [
    ("short", "What is 2+2?"),
    ("medium", "Explain the difference between supervised and unsupervised machine learning in 2-3 sentences."),
    ("long", "Write a detailed explanation of how neural networks work, covering: 1) perceptrons, 2) layers, 3) activation functions, 4) backpropagation, and 5) gradient descent."),
]

MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]


class TokenTrackingExperiment(BaseExperiment):
    def run(self) -> list[ExperimentResult]:
        results = []
        for prompt_name, prompt in PROMPT_VARIANTS:
            for model in MODELS:
                result = self.run_single(
                    run_index=1,
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    extra_params={"prompt_length": prompt_name},
                )
                results.append(result)
        return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_experiments.py -v
```

**Step 5: Commit**

```bash
git add src/experiments/token_tracking.py tests/test_experiments.py
git commit -m "feat: add token tracking experiment"
```

---

### Task 10: experiments/streaming.py

**Files:**
- Create: `src/experiments/streaming.py`
- Modify: `tests/test_experiments.py`

**Step 1: Add failing tests**

```python
# append to tests/test_experiments.py
from src.experiments.streaming import StreamingExperiment

@pytest.fixture
def mock_streaming_client():
    client = MagicMock()
    client.call.return_value = {
        "response_text": "batch response",
        "input_tokens": 50,
        "output_tokens": 30,
        "cost_usd": 0.001,
        "latency_ms": 800,
    }
    client.stream.return_value = {
        "response_text": "streaming response",
        "input_tokens": 50,
        "output_tokens": 30,
        "cost_usd": 0.001,
        "latency_ms": 820,
        "ttft_ms": 120,
    }
    return client

def test_streaming_returns_two_results(mock_streaming_client, storage):
    config = ExperimentConfig(name="streaming", prompt="test")
    exp = StreamingExperiment(mock_streaming_client, storage, config)
    results = exp.run()
    assert len(results) == 2

def test_streaming_tags_mode(mock_streaming_client, storage):
    config = ExperimentConfig(name="streaming", prompt="test")
    exp = StreamingExperiment(mock_streaming_client, storage, config)
    results = exp.run()
    modes = {r.params["mode"] for r in results}
    assert modes == {"batch", "stream"}

def test_streaming_result_has_ttft(mock_streaming_client, storage):
    config = ExperimentConfig(name="streaming", prompt="test")
    exp = StreamingExperiment(mock_streaming_client, storage, config)
    results = exp.run()
    stream_result = next(r for r in results if r.params["mode"] == "stream")
    assert stream_result.ttft_ms == 120
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_experiments.py::test_streaming_returns_two_results -v
```

**Step 3: Write implementation**

```python
# src/experiments/streaming.py
from src.experiments.base import BaseExperiment
from src.config import ExperimentResult


class StreamingExperiment(BaseExperiment):
    def run(self) -> list[ExperimentResult]:
        results = []

        batch_raw = self.client.call(
            model=self.config.model,
            prompt=self.config.prompt,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        batch_result = ExperimentResult(
            experiment=self.config.name,
            model=self.config.model,
            prompt=self.config.prompt,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            run_index=1,
            params={"mode": "batch"},
            **batch_raw,
        )
        self.storage.save(batch_result)
        results.append(batch_result)

        stream_raw = self.client.stream(
            model=self.config.model,
            prompt=self.config.prompt,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        stream_result = ExperimentResult(
            experiment=self.config.name,
            model=self.config.model,
            prompt=self.config.prompt,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            run_index=2,
            params={"mode": "stream"},
            **stream_raw,
        )
        self.storage.save(stream_result)
        results.append(stream_result)

        return results
```

**Step 4: Run all tests to verify they pass**

```bash
pytest -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add src/experiments/streaming.py tests/test_experiments.py
git commit -m "feat: add streaming experiment"
```

---

## Phase 4: Visualisation & UI

### Task 11: visualisation.py — Plotly charts

**Files:**
- Create: `src/visualisation.py`

No TDD here — chart output is visual. Verify manually by running the Streamlit app in Task 12.

```python
# src/visualisation.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.config import ExperimentResult
from src.experiments.temperature import TemperatureSweepExperiment


def plot_temperature_sweep(results: list[ExperimentResult]) -> go.Figure:
    by_temp: dict[float, list[str]] = {}
    for r in results:
        key = r.temperature or 0.0
        by_temp.setdefault(key, []).append(r.response_text)

    temps = sorted(by_temp.keys())
    similarities = [TemperatureSweepExperiment.similarity(by_temp[t]) for t in temps]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temps, y=similarities, mode="lines+markers", name="Similarity"))
    fig.update_layout(
        title="Temperature vs Response Consistency",
        xaxis_title="Temperature",
        yaxis_title="Similarity Score (0–1)",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def plot_system_prompts(results: list[ExperimentResult]) -> go.Figure:
    rows = [
        {
            "variant": r.params.get("variant", "unknown"),
            "response_length": len(r.response_text),
            "output_tokens": r.output_tokens,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="variant",
        y=["response_length", "output_tokens"],
        barmode="group",
        title="System Prompt Variants: Response Length vs Output Tokens",
    )
    return fig


def plot_model_comparison(results: list[ExperimentResult]) -> go.Figure:
    rows = [
        {
            "model": r.model.split("-")[1],
            "latency_ms": r.latency_ms,
            "cost_usd_per_1k": round(r.cost_usd * 1000, 4),
            "output_tokens": r.output_tokens,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="model",
        y=["latency_ms", "cost_usd_per_1k", "output_tokens"],
        barmode="group",
        title="Model Comparison: Latency, Cost (per 1k requests), Output Tokens",
    )
    return fig


def plot_token_usage(results: list[ExperimentResult]) -> go.Figure:
    rows = [
        {
            "label": f"{r.params.get('prompt_length', '?')} / {r.model.split('-')[1]}",
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Input Tokens", x=df["label"], y=df["input_tokens"]))
    fig.add_trace(go.Bar(name="Output Tokens", x=df["label"], y=df["output_tokens"]))
    fig.update_layout(barmode="stack", title="Token Usage by Prompt Length and Model")
    return fig


def plot_streaming(results: list[ExperimentResult]) -> go.Figure:
    batch = next((r for r in results if r.params.get("mode") == "batch"), None)
    stream = next((r for r in results if r.params.get("mode") == "stream"), None)
    if not batch or not stream:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Batch — Total", x=["Total Time (ms)"], y=[batch.latency_ms]))
    fig.add_trace(go.Bar(name="Stream — Total", x=["Total Time (ms)"], y=[stream.latency_ms]))
    fig.add_trace(go.Bar(name="Stream — TTFT", x=["Time to First Token (ms)"], y=[stream.ttft_ms or 0]))
    fig.update_layout(barmode="group", title="Streaming vs Batch: Latency")
    return fig
```

**Commit**

```bash
git add src/visualisation.py
git commit -m "feat: add Plotly chart generators"
```

---

### Task 12: app.py — Streamlit UI

**Files:**
- Create: `app.py`

```python
# app.py
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.client import ClaudeClient
from src.storage import SQLiteStorage
from src.config import ExperimentConfig, MODEL_PRICING
from src.experiments.temperature import TemperatureSweepExperiment
from src.experiments.system_prompts import SystemPromptsExperiment
from src.experiments.model_compare import ModelCompareExperiment
from src.experiments.token_tracking import TokenTrackingExperiment
from src.experiments.streaming import StreamingExperiment
from src import visualisation

load_dotenv()

st.set_page_config(page_title="Claude API Explorer", layout="wide")

COST_CAPS = {"per_run": 0.10, "session_warn": 1.00}
SESSION_COST_KEY = "session_cost_usd"
if SESSION_COST_KEY not in st.session_state:
    st.session_state[SESSION_COST_KEY] = 0.0

EXPERIMENT_MAP = {
    "Temperature Sweep": "temperature_sweep",
    "System Prompts": "system_prompts",
    "Model Comparison": "model_compare",
    "Token Economics": "token_tracking",
    "Streaming": "streaming",
}

COST_ESTIMATES = {
    "Temperature Sweep": "~$0.02 (6 temps × 3 runs, Sonnet)",
    "System Prompts": "~$0.01 (5 variants, Sonnet)",
    "Model Comparison": "~$0.005 (Haiku + Sonnet, opt-in Opus adds ~$0.03)",
    "Token Economics": "~$0.01 (3 prompts × 2 models)",
    "Streaming": "~$0.003 (1 batch + 1 stream, Sonnet)",
}


@st.cache_resource
def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        st.stop()
    return ClaudeClient(api_key)


@st.cache_resource
def get_storage():
    return SQLiteStorage()


client = get_client()
storage = get_storage()

st.title("Claude API Explorer")

tab_run, tab_history, tab_posts = st.tabs(["Run Experiment", "History", "Post Content"])

with tab_run:
    with st.sidebar:
        st.header("Configure")
        experiment_type = st.selectbox("Experiment", list(EXPERIMENT_MAP.keys()))
        prompt = st.text_area(
            "Prompt",
            value="Explain what machine learning is in simple terms.",
            height=100,
        )
        num_runs = 3
        include_opus = False

        if experiment_type == "Temperature Sweep":
            num_runs = st.slider("Runs per temperature", 1, 5, 3)
        elif experiment_type == "Model Comparison":
            include_opus = st.checkbox("Include Opus (adds ~$0.03)", value=False)

        st.info(f"Estimated cost: {COST_ESTIMATES[experiment_type]}")
        if st.session_state[SESSION_COST_KEY] >= COST_CAPS["session_warn"]:
            st.warning(f"Session spend: ${st.session_state[SESSION_COST_KEY]:.4f} (over $1.00 warning threshold)")
        else:
            st.caption(f"Session spend: ${st.session_state[SESSION_COST_KEY]:.4f}")

    config = ExperimentConfig(
        name=EXPERIMENT_MAP[experiment_type],
        prompt=prompt,
        num_runs=num_runs,
    )

    if st.button("Run Experiment", type="primary"):
        with st.spinner("Running…"):
            try:
                if experiment_type == "Temperature Sweep":
                    exp = TemperatureSweepExperiment(client, storage, config)
                elif experiment_type == "System Prompts":
                    exp = SystemPromptsExperiment(client, storage, config)
                elif experiment_type == "Model Comparison":
                    exp = ModelCompareExperiment(client, storage, config, include_opus=include_opus)
                elif experiment_type == "Token Economics":
                    exp = TokenTrackingExperiment(client, storage, config)
                else:
                    exp = StreamingExperiment(client, storage, config)

                results = exp.run()
                total_cost = sum(r.cost_usd for r in results)
                st.session_state[SESSION_COST_KEY] += total_cost

                st.success(f"Done — {len(results)} runs, ${total_cost:.4f} spent")

                if experiment_type == "Temperature Sweep":
                    st.plotly_chart(visualisation.plot_temperature_sweep(results), use_container_width=True)
                elif experiment_type == "System Prompts":
                    st.plotly_chart(visualisation.plot_system_prompts(results), use_container_width=True)
                elif experiment_type == "Model Comparison":
                    st.plotly_chart(visualisation.plot_model_comparison(results), use_container_width=True)
                elif experiment_type == "Token Economics":
                    st.plotly_chart(visualisation.plot_token_usage(results), use_container_width=True)
                else:
                    st.plotly_chart(visualisation.plot_streaming(results), use_container_width=True)

                with st.expander("Raw responses"):
                    for r in results:
                        st.markdown(f"**Run {r.run_index} | temp={r.temperature} | {r.model}**")
                        st.text(r.response_text[:500])

            except Exception as e:
                st.error(f"Error: {e}")

with tab_history:
    all_results = storage.get_all()
    if not all_results:
        st.info("No results yet. Run an experiment first.")
    else:
        df = pd.DataFrame([
            {
                "experiment": r.experiment,
                "model": r.model,
                "timestamp": r.timestamp,
                "temperature": r.temperature,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "output_tokens": r.output_tokens,
            }
            for r in all_results
        ])

        exp_options = ["All"] + sorted(df["experiment"].unique().tolist())
        exp_filter = st.selectbox("Filter by experiment", exp_options)
        if exp_filter != "All":
            df = df[df["experiment"] == exp_filter]

        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="results.csv", mime="text/csv")

with tab_posts:
    st.header("Post Content Generator")
    st.caption("Auto-generated findings for each experiment. Copy and paste into your posts.")

    selected = st.selectbox("Select experiment", list(EXPERIMENT_MAP.values()))
    exp_results = storage.get_by_experiment(selected)

    if not exp_results:
        st.info(f"No results for '{selected}'. Run this experiment first.")
    else:
        if selected == "temperature_sweep":
            from src.experiments.temperature import TemperatureSweepExperiment
            by_temp: dict = {}
            for r in exp_results:
                key = r.temperature or 0.0
                by_temp.setdefault(key, []).append(r.response_text)
            low = TemperatureSweepExperiment.similarity(by_temp.get(0.0, ["x"]))
            high = TemperatureSweepExperiment.similarity(by_temp.get(1.0, ["x"]))
            st.code(
                f"At temperature 0.0, response similarity was {low:.0%}. "
                f"At temperature 1.0, it dropped to {high:.0%}.",
                language=None,
            )

        elif selected == "system_prompts":
            none_result = next((r for r in exp_results if r.params.get("variant") == "none"), None)
            best_result = max(exp_results, key=lambda r: r.output_tokens)
            if none_result and best_result:
                ratio = best_result.output_tokens / max(none_result.output_tokens, 1)
                st.code(
                    f"With no system prompt: {none_result.output_tokens} output tokens. "
                    f"With '{best_result.params.get('variant')}': {best_result.output_tokens} tokens "
                    f"({ratio:.1f}x more structured output).",
                    language=None,
                )

        elif selected == "model_compare":
            haiku = next((r for r in exp_results if "haiku" in r.model), None)
            sonnet = next((r for r in exp_results if "sonnet" in r.model), None)
            if haiku and sonnet:
                cost_ratio = sonnet.cost_usd / max(haiku.cost_usd, 1e-9)
                latency_ratio = sonnet.latency_ms / max(haiku.latency_ms, 1)
                st.code(
                    f"Haiku: ${haiku.cost_usd:.5f} per request, {haiku.latency_ms}ms. "
                    f"Sonnet: ${sonnet.cost_usd:.5f} ({cost_ratio:.1f}x cost), {sonnet.latency_ms}ms ({latency_ratio:.1f}x latency).",
                    language=None,
                )

        elif selected == "token_tracking":
            short = next((r for r in exp_results if r.params.get("prompt_length") == "short"), None)
            long_ = next((r for r in exp_results if r.params.get("prompt_length") == "long"), None)
            if short and long_:
                st.code(
                    f"Short prompt: {short.input_tokens} input tokens, ${short.cost_usd:.5f}. "
                    f"Long prompt: {long_.input_tokens} input tokens, ${long_.cost_usd:.5f} "
                    f"({long_.input_tokens / max(short.input_tokens, 1):.1f}x more tokens).",
                    language=None,
                )

        elif selected == "streaming":
            batch = next((r for r in exp_results if r.params.get("mode") == "batch"), None)
            stream = next((r for r in exp_results if r.params.get("mode") == "stream"), None)
            if batch and stream:
                st.code(
                    f"Batch total time: {batch.latency_ms}ms. "
                    f"Streaming total time: {stream.latency_ms}ms. "
                    f"Time to first token (streaming): {stream.ttft_ms}ms — "
                    f"first output in {stream.ttft_ms / max(batch.latency_ms, 1):.0%} of batch total time.",
                    language=None,
                )

        st.plotly_chart(_get_chart(selected, exp_results), use_container_width=True)


def _get_chart(experiment: str, results):
    if experiment == "temperature_sweep":
        return visualisation.plot_temperature_sweep(results)
    elif experiment == "system_prompts":
        return visualisation.plot_system_prompts(results)
    elif experiment == "model_compare":
        return visualisation.plot_model_comparison(results)
    elif experiment == "token_tracking":
        return visualisation.plot_token_usage(results)
    elif experiment == "streaming":
        return visualisation.plot_streaming(results)
    return go.Figure()
```

**Step to verify: run the app**

```bash
streamlit run app.py
```

Expected: app opens at `http://localhost:8501`, all 3 tabs visible, sidebar controls update per experiment.

**Commit**

```bash
git add app.py src/visualisation.py
git commit -m "feat: add Streamlit UI with run, history, and post content tabs"
```

---

## Phase 5: Polish

### Task 13: README.md

**Files:**
- Create: `README.md`

Write a README with these sections:

1. **Project description** — one paragraph, what it does and why
2. **Setup** — exact commands to clone, create `.env`, install, run
3. **Experiments** — one-line description of each of the 5 experiments
4. **Key findings** — leave placeholder text like `[screenshot: temperature chart]` to fill in after running
5. **Tech stack** — table (already in design doc)

Keep it under 150 lines. No fluff.

**Setup section must include:**

```bash
git clone <your-repo-url>
cd claude-api-explorer
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
streamlit run app.py
```

**Commit**

```bash
git add README.md
git commit -m "docs: add README with setup instructions and experiment overview"
```

---

## Final Verification

Run the full test suite:

```bash
pytest -v
```

Expected: all tests PASS, no warnings.

Run the app end-to-end:

```bash
streamlit run app.py
```

- Run Temperature Sweep experiment
- Check History tab shows results
- Check Post Content tab shows copyable finding

---

## Definition of Done

- [ ] `pytest -v` — all tests pass
- [ ] `streamlit run app.py` — app runs locally
- [ ] All 5 experiments produce results and charts
- [ ] History tab shows past runs and CSV download works
- [ ] Post Content tab shows copyable stats for each experiment
- [ ] README has setup instructions
- [ ] `.env` is gitignored, `.env.example` is committed
