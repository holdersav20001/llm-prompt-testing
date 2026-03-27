import pytest
from unittest.mock import MagicMock
from src.experiments.streaming import StreamingExperiment
from src.config import ExperimentConfig
from src.storage import SQLiteStorage

@pytest.fixture
def mock_client():
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

@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(db_path=str(tmp_path / "test.db"))

def test_streaming_returns_two_results(mock_client, storage):
    config = ExperimentConfig(name="streaming", prompt="test")
    exp = StreamingExperiment(mock_client, storage, config)
    results = exp.run()
    assert len(results) == 2

def test_streaming_tags_mode(mock_client, storage):
    config = ExperimentConfig(name="streaming", prompt="test")
    exp = StreamingExperiment(mock_client, storage, config)
    results = exp.run()
    modes = {r.params["mode"] for r in results}
    assert modes == {"batch", "stream"}

def test_streaming_result_has_ttft(mock_client, storage):
    config = ExperimentConfig(name="streaming", prompt="test")
    exp = StreamingExperiment(mock_client, storage, config)
    results = exp.run()
    stream_result = next(r for r in results if r.params["mode"] == "stream")
    assert stream_result.ttft_ms == 120
