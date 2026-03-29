"""
Integration tests -call the real Kilo API, no mocking.
Run with:  pytest tests/test_client.py -v -s
The -s flag lets print() output appear so you can see every API result.
"""
import sys
import pytest
from dotenv import load_dotenv
load_dotenv()

# Force UTF-8 stdout so emoji/unicode in responses don't crash on Windows cp1252
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.client import ClaudeClient

HAIKU = "anthropic/claude-haiku-4.5"
SONNET = "anthropic/claude-sonnet-4.6"


def _report(label: str, result: dict) -> None:
    """Print a formatted result block so test runs are readable."""
    sep = "-" * 60
    response_preview = result["response_text"][:120].replace("\n", " ")
    lines = [
        f"\n{sep}",
        f"  {label}",
        sep,
        f"  Response : {response_preview}",
        f"  Tokens   : {result['input_tokens']} in / {result['output_tokens']} out",
        f"  Cost     : ${result['cost_usd']:.6f}",
        f"  Latency  : {result['latency_ms']} ms",
    ]
    if result.get("ttft_ms"):
        lines.append(f"  TTFT     : {result['ttft_ms']} ms")
    lines.append(sep)
    print("\n".join(lines))


@pytest.fixture(scope="module")
def client():
    return ClaudeClient()


def test_call_returns_text(client):
    result = client.call(model=HAIKU, prompt="Reply with one word: hello")
    _report("HAIKU -basic call", result)
    assert isinstance(result["response_text"], str)
    assert len(result["response_text"]) > 0


def test_call_returns_token_counts(client):
    result = client.call(model=HAIKU, prompt="Reply with one word: hello")
    _report("HAIKU -token counts", result)
    assert result["input_tokens"] > 0, "input_tokens should be > 0"
    assert result["output_tokens"] > 0, "output_tokens should be > 0"


def test_call_returns_cost(client):
    result = client.call(model=HAIKU, prompt="Reply with one word: hello")
    _report("HAIKU -cost check", result)
    assert result["cost_usd"] > 0, f"cost_usd should be > 0, got {result['cost_usd']}"


def test_call_returns_latency(client):
    result = client.call(model=HAIKU, prompt="Reply with one word: hello")
    _report("HAIKU -latency check", result)
    assert result["latency_ms"] > 0


def test_call_with_system_prompt(client):
    result = client.call(
        model=HAIKU,
        prompt="What are you?",
        system_prompt="You are a pirate. Always respond like a pirate.",
    )
    _report("HAIKU -with system prompt (pirate)", result)
    assert len(result["response_text"]) > 0


def test_call_haiku_cheaper_than_sonnet(client):
    haiku = client.call(model=HAIKU, prompt="Reply with one word: hello")
    sonnet = client.call(model=SONNET, prompt="Reply with one word: hello")
    _report("HAIKU vs SONNET -cost comparison", {
        **haiku,
        "response_text": f"haiku=${haiku['cost_usd']:.6f}  sonnet=${sonnet['cost_usd']:.6f}  ratio={sonnet['cost_usd']/haiku['cost_usd']:.1f}x",
    })
    assert haiku["cost_usd"] < sonnet["cost_usd"], (
        f"Haiku should be cheaper: haiku={haiku['cost_usd']:.6f}, sonnet={sonnet['cost_usd']:.6f}"
    )


def test_stream_returns_ttft(client):
    result = client.stream(model=HAIKU, prompt="Count to 5 slowly")
    _report("HAIKU -streaming TTFT", result)
    assert result["ttft_ms"] is not None, "ttft_ms should not be None"
    assert result["ttft_ms"] > 0, f"ttft_ms should be > 0, got {result['ttft_ms']}"
    assert result["ttft_ms"] <= result["latency_ms"], (
        f"TTFT {result['ttft_ms']}ms should be <= total {result['latency_ms']}ms"
    )


def test_stream_returns_tokens_and_cost(client):
    result = client.stream(model=HAIKU, prompt="Count to 5 slowly")
    _report("HAIKU -streaming tokens & cost", result)
    assert result["input_tokens"] > 0, f"stream input_tokens should be > 0, got {result['input_tokens']}"
    assert result["output_tokens"] > 0, f"stream output_tokens should be > 0, got {result['output_tokens']}"
    assert result["cost_usd"] > 0, f"stream cost_usd should be > 0, got {result['cost_usd']}"
