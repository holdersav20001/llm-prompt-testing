import time
import anthropic
from anthropic import APIStatusError
from src.config import calculate_cost


class ClaudeClient:
    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
        **kwargs,
    ) -> dict:
        messages = [{"role": "user", "content": prompt}]
        api_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if system_prompt:
            api_kwargs["system"] = system_prompt

        for attempt in range(3):
            try:
                start = time.time()
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    **api_kwargs,
                )
                latency_ms = int((time.time() - start) * 1000)
                return {
                    "response_text": response.content[0].text,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cost_usd": calculate_cost(model, response.usage.input_tokens, response.usage.output_tokens),
                    "latency_ms": latency_ms,
                }
            except APIStatusError as e:
                if e.status_code == 401:
                    raise
                if e.status_code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    def stream(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> dict:
        messages = [{"role": "user", "content": prompt}]
        api_kwargs = {}
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if system_prompt:
            api_kwargs["system"] = system_prompt

        ttft_ms = None
        collected = []
        start = time.time()

        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            **api_kwargs,
        ) as stream:
            for text in stream.text_stream:
                if ttft_ms is None:
                    ttft_ms = int((time.time() - start) * 1000)
                collected.append(text)
            final = stream.get_final_message()

        latency_ms = int((time.time() - start) * 1000)
        return {
            "response_text": "".join(collected),
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
            "cost_usd": calculate_cost(model, final.usage.input_tokens, final.usage.output_tokens),
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
        }
