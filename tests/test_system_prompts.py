import pytest
from unittest.mock import MagicMock
from src.experiments.system_prompts import SystemPromptsExperiment, SYSTEM_PROMPT_VARIANTS
from src.config import ExperimentConfig
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

def test_system_prompts_uses_fixed_temperature(mock_client, storage):
    config = ExperimentConfig(name="system_prompts", prompt="test")
    exp = SystemPromptsExperiment(mock_client, storage, config)
    results = exp.run()
    assert all(r.temperature == 0.0 for r in results)
