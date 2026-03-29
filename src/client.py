import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _make_openai_client() -> OpenAI:
    api_key = os.environ.get("KILO_API_KEY")
    if not api_key:
        raise RuntimeError("KILO_API_KEY not set in environment")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.kilo.ai/api/gateway",
    )


class ClaudeClient:
    def __init__(self):
        self._client = _make_openai_client()

    def call(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
        **kwargs,
    ) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        start = time.time()
        response = self._client.chat.completions.create(**call_kwargs)
        latency_ms = int((time.time() - start) * 1000)

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost_usd = _estimate_cost(model, input_tokens, output_tokens)

        content = response.choices[0].message.content if response.choices else None
        if not content:
            import sys
            print(f"\n=== Empty/None content from {model} (finish_reason={response.choices[0].finish_reason if response.choices else 'no choices'}) ===", file=sys.stderr)
            print(response.model_dump_json(indent=2), file=sys.stderr)
            content = f"[no content returned — finish_reason: {response.choices[0].finish_reason if response.choices else 'no choices'}]"

        return {
            "response_text": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        }

    def stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        ttft_ms = None
        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0
        start = time.time()

        with self._client.chat.completions.create(**call_kwargs) as stream:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if ttft_ms is None:
                        ttft_ms = int((time.time() - start) * 1000)
                    collected.append(content)
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0

        latency_ms = int((time.time() - start) * 1000)
        cost_usd = _estimate_cost(model, input_tokens, output_tokens)

        return {
            "response_text": "".join(collected),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
        }


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    from src.config import MODEL_PRICING
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # Unknown model — use sonnet as a rough fallback
        pricing = MODEL_PRICING.get("anthropic/claude-sonnet-4.6", {"input": 3.00, "output": 15.00})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
