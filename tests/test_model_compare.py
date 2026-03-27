import pytest
from unittest.mock import MagicMock
from src.experiments.model_compare import ModelCompareExperiment
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

def test_model_compare_uses_fixed_temperature(mock_client, storage):
    config = ExperimentConfig(name="model_compare", prompt="test")
    exp = ModelCompareExperiment(mock_client, storage, config)
    results = exp.run()
    assert all(r.temperature == 0.0 for r in results)
