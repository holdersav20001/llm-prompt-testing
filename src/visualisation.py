import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.config import ExperimentResult
from src.experiments.temperature import TemperatureSweepExperiment


def plot_temperature_sweep(results: list[ExperimentResult]) -> go.Figure:
    by_temp: dict[float, list[str]] = {}
    for r in results:
        key = r.temperature or 0.0
        by_temp.setdefault(key, []).append(r.response_text)

    temps = sorted(by_temp.keys())
    similarities = [TemperatureSweepExperiment.similarity(by_temp[t]) for t in temps]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temps, y=similarities, mode="lines+markers", name="Similarity"))
    fig.update_layout(
        title="Temperature vs Response Consistency",
        xaxis_title="Temperature",
        yaxis_title="Similarity Score (0–1)",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def plot_system_prompts(results: list[ExperimentResult]) -> go.Figure:
    rows = [
        {
            "variant": r.params.get("variant", "unknown"),
            "response_length": len(r.response_text),
            "output_tokens": r.output_tokens,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="variant",
        y=["response_length", "output_tokens"],
        barmode="group",
        title="System Prompt Variants: Response Length vs Output Tokens",
    )
    return fig


def plot_model_comparison(results: list[ExperimentResult]) -> go.Figure:
    rows = [
        {
            "model": r.model.split("-")[1],
            "latency_ms": r.latency_ms,
            "cost_usd_per_1k": round(r.cost_usd * 1000, 4),
            "output_tokens": r.output_tokens,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="model",
        y=["latency_ms", "cost_usd_per_1k", "output_tokens"],
        barmode="group",
        title="Model Comparison: Latency, Cost (per 1k requests), Output Tokens",
    )
    return fig


def plot_token_usage(results: list[ExperimentResult]) -> go.Figure:
    rows = [
        {
            "label": f"{r.params.get('prompt_length', '?')} / {r.model.split('-')[1]}",
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Input Tokens", x=df["label"], y=df["input_tokens"]))
    fig.add_trace(go.Bar(name="Output Tokens", x=df["label"], y=df["output_tokens"]))
    fig.update_layout(barmode="stack", title="Token Usage by Prompt Length and Model")
    return fig


def plot_streaming(results: list[ExperimentResult]) -> go.Figure:
    batch = next((r for r in results if r.params.get("mode") == "batch"), None)
    stream = next((r for r in results if r.params.get("mode") == "stream"), None)
    if not batch or not stream:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Batch — Total", x=["Total Time (ms)"], y=[batch.latency_ms]))
    fig.add_trace(go.Bar(name="Stream — Total", x=["Total Time (ms)"], y=[stream.latency_ms]))
    fig.add_trace(go.Bar(name="Stream — TTFT", x=["Time to First Token (ms)"], y=[stream.ttft_ms or 0]))
    fig.update_layout(barmode="group", title="Streaming vs Batch: Latency")
    return fig
