import pytest
from src.storage import SQLiteStorage
from src.config import ExperimentResult

@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(db_path=str(tmp_path / "test.db"))

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
