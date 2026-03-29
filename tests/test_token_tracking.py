import pytest
from unittest.mock import MagicMock
from src.experiments.token_tracking import TokenTrackingExperiment, PROMPT_VARIANTS
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

def test_token_tracking_uses_both_models(mock_client, storage):
    config = ExperimentConfig(name="token_tracking", prompt="ignored")
    exp = TokenTrackingExperiment(mock_client, storage, config)
    results = exp.run()
    models = {r.model for r in results}
    assert "anthropic/claude-haiku-4.5" in models
    assert "anthropic/claude-sonnet-4.6" in models
