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
    assert abs(cost - 4.80) < 0.001

def test_calculate_cost_sonnet():
    cost = calculate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(cost - 18.00) < 0.001

def test_experiment_config_defaults():
    config = ExperimentConfig(name="my_test", prompt="what is 2+2?")
    assert config.model == "claude-sonnet-4-6"
    assert config.num_runs == 3
    assert config.max_tokens == 1024
