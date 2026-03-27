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
