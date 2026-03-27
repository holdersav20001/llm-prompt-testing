import pytest
from unittest.mock import MagicMock, patch
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
    mocker.patch.object(
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
