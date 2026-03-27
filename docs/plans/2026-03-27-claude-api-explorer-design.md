# Claude API Explorer — Design Document

**Date:** 2026-03-27
**Approach:** Content-led (Option B) — 5 experiments mapped directly to 5 blog posts

---

## Goal

Build a Python/Streamlit application that systematically tests Claude API parameters, stores results in SQLite, and visualises findings with Plotly. Serves two purposes: a portfolio piece (GitHub repo + README + screenshots) and a content engine (each experiment feeds one blog post).

---

## The 5 Experiments

| Experiment | Post angle | What it measures |
|---|---|---|
| Temperature sweep | "I tested every Claude API param so you don't have to" | Response consistency vs creativity at 0.0→1.0 in 0.2 steps, 3 runs each |
| System prompts | "Your system prompt is doing 10% of what it could" | 5 variants: none, short role, detailed role, format constraints, persona+constraints |
| Model comparison | "Haiku vs Sonnet vs Opus: when to use which" | Quality, latency, cost across 3 models on the same prompts |
| Token economics | "Here's exactly what each model costs per task" | Input/output token counts, cost per request, prompt caching effect |
| Streaming | "Why streaming doesn't make Claude faster but feels instant" | Time-to-first-token vs total time, batch vs stream |

Dropped from original spec: sampling grid (too expensive, no dedicated post) and structured output (footnote in temperature post).

---

## Architecture

```
Streamlit UI
    ↓ user picks experiment + params
Experiment Runner (one module per experiment)
    ↓
Claude API Client (retry, cost tracking, streaming)
    ↓
Claude API
    ↓
SQLite Storage + Plotly Visualisation
    ↓
Streamlit UI (results + charts)
```

### Repo Structure

```
claude-api-explorer/
├── app.py
├── src/
│   ├── client.py
│   ├── config.py
│   ├── storage.py
│   ├── visualisation.py
│   └── experiments/
│       ├── base.py
│       ├── temperature.py
│       ├── system_prompts.py
│       ├── model_compare.py
│       ├── token_tracking.py
│       └── streaming.py
├── data/
│   └── .gitkeep
├── tests/
│   ├── test_client.py
│   ├── test_storage.py
│   └── test_experiments.py
├── docs/
│   └── plans/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Data Model

```python
class ExperimentResult(BaseModel):
    id: str                    # uuid
    experiment: str            # "temperature", "model_compare", etc.
    timestamp: datetime
    model: str
    prompt: str
    system_prompt: str
    response_text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float            # calculated at save time
    latency_ms: int
    ttft_ms: int | None        # streaming only
    temperature: float | None
    params: dict               # any extra params
    run_index: int             # repeat number (1, 2, 3...)
```

SQLite table has `experiment`, `model`, `temperature`, `cost_usd`, `latency_ms` as real columns. Everything else in `config_json` / `result_json`.

### Model Pricing (per 1M tokens)

| Model | Input | Output |
|---|---|---|
| claude-haiku-4-5-20251001 | $0.80 | $4.00 |
| claude-sonnet-4-6 | $3.00 | $15.00 |
| claude-opus-4-6 | $15.00 | $75.00 |

---

## UI Design

Three tabs:

**Tab 1 — Run Experiment**
- Sidebar: experiment selector + relevant parameter controls
- Cost estimate before running
- Progress bar during run
- Results: response samples, key metrics, Plotly chart

**Tab 2 — History**
- Table of past runs, filterable by experiment and model
- Click row to see full results
- Download CSV

**Tab 3 — Post Content**
- Per-experiment findings panel with auto-generated copyable stats
- Feeds blog posts directly without manual data extraction

### Charts

| Experiment | Chart type |
|---|---|
| Temperature sweep | Line: temperature vs similarity score |
| System prompts | Bar: response length + format adherence by variant |
| Model comparison | Grouped bar: latency / cost / tokens per model |
| Token economics | Stacked bar: input vs output tokens; cost per request |
| Streaming | Side-by-side bar: TTFT vs total time |

---

## Error Handling & Cost Guardrails

**API errors:**
- 429 rate limit: exponential backoff, max 3 retries
- 401 auth: fail immediately, clear UI message
- Timeout: 30s default, failed runs saved to DB (not silently dropped)

**Cost guardrails:**
- Cost estimate shown before every run
- Per-run hard cap: $0.10 (configurable)
- Per-session soft cap: warn at $1.00
- Model comparison: Haiku + Sonnet by default; Opus is opt-in

**Tests:**
- `test_client.py`: mock API call, retry on 429, cost calculation
- `test_storage.py`: save + retrieve, verify queryable columns
- `test_experiments.py`: each experiment with mock client, verify result shape
