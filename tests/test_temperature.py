import pytest
from unittest.mock import MagicMock
from src.experiments.temperature import TemperatureSweepExperiment
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
