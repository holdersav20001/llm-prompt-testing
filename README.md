# LLM Prompt Testing

A Streamlit application for running structured experiments across 30+ LLM models via the [Kilo AI](https://kilo.ai) gateway. Design prompts, configure system prompts, select models and temperatures, run them in parallel, and score the results with an AI reviewer — all from a single UI.

---

## What It Does

The app lets you build a **matrix experiment**: pick any combination of models, system prompts, prompt variants, and temperatures, then fire them all in parallel. Results are stored in SQLite and displayed as scrollable run cards with metrics tables. An AI reviewer can score every response 1–10 across six engineering criteria and produce a ranked report.

**Key features:**
- Live model list fetched from the Kilo AI gateway (330+ models across 20+ providers)
- Parallel execution with per-task status tracking
- AI-powered scoring: Layout, Testability, Performance, Exception Management, Readability, Completeness (summed out of 60)
- Compare tab: pick responses from any past run and have an AI compare them without re-running
- Full SQLite persistence for results, reviews, and comparisons

---

## Setup

```bash
git clone https://github.com/holdersav20001/llm-prompt-testing.git
cd llm-prompt-testing
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
# source venv/bin/activate      # Mac/Linux
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add your KILO_API_KEY
streamlit run app.py --server.headless true
```

---

## Project Structure

```
llm-prompt-testing/
├── app.py                        # Streamlit UI — all tabs, review, compare
├── requirements.txt
├── src/
│   ├── client.py                 # OpenAI-compatible client → Kilo gateway
│   ├── config.py                 # Model pricing, ExperimentResult, Variant models
│   ├── storage.py                # SQLite: results, variants, reviews, comparisons
│   ├── visualisation.py          # Plotly charts
│   └── experiments/
│       ├── base.py               # BaseExperiment
│       ├── matrix.py             # Parallel matrix runner (ThreadPoolExecutor)
│       ├── model_compare.py
│       ├── streaming.py
│       ├── system_prompts.py
│       ├── temperature.py
│       └── token_tracking.py
├── docs/
│   └── model-report-2026-03-28.md   # Full benchmark report
└── tests/
```

---

## Benchmark: AWS Lambda Code Generation

The main experiment series tested 13 LLM responses to the same code generation task, reviewed and scored by AI (OpenAI o4-mini and Claude Sonnet 4.6).

### Prompt

```
Write a Lambda function that reads from Kafka and writes to an S3 file.
Must be efficient, include error handling, instrumentation and appropriate logging.
```

### System Prompt

```
# Senior AWS Developer — Kafka · Lambda · S3
Code must be testable, performant, modularised, not a big chunk of code,
easy to understand and performant.
```

### Results (scored out of 60)

| Rank | Model | Layout | Testability | Performance | Exc. Mgmt | Readability | Completeness | Total |
|------|-------|--------|-------------|-------------|-----------|-------------|--------------|-------|
| 1 | **gpt-5.4** | 10 | 10 | 10 | 10 | 10 | 9 | **59** |
| 2 | **claude-opus-4.6** | 9 | 10 | 8 | 9 | 10 | 7 | **53** |
| 3 | **claude-sonnet-4.6** | 9 | 10 | 9 | 8 | 9 | 7 | **52** |
| 4 | gemini-3.1-pro-preview | 8 | 8 | 7 | 8 | 9 | 8 | **48** |
| 5 | minimax-m2.7 | 9 | 8 | 8 | 5 | 7 | 7 | **44** |
| 6 | qwen3-235b-a22b-thinking | 7 | 6 | 7 | 7 | 7 | 7 | **41** |
| 7 | kimi-k2-thinking | 8 | 7 | 6 | 5 | 8 | 6 | **40** |
| 8 | minimax-m2.5 | 9 | 6 | 5 | 5 | 7 | 6 | **38** |
| 9 | deepseek-v3.2 | 7 | 4 | 6 | 5 | 7 | 6 | **35** |
| 10 | glm-4.7 | 8 | 4 | 5 | 5 | 6 | 6 | **34** |
| 11 | nemotron-ultra-253b | 6 | 3 | 6 | 5 | 5 | 5 | **30** |
| 12 | nemotron-3-super-120b | 6 | 2 | 2 | 4 | 5 | 4 | **23** |
| 13 | llama-4-maverick | 4 | 2 | 2 | 2 | 5 | 2 | **17** |

*Reviewer: OpenAI o4-mini (runs 1–11), Claude Sonnet 4.6 (runs 12–13)*

---

## Key Findings

### 1. System prompt specificity beats model size

With a generic "Senior Developer" system prompt, Claude Sonnet 4.6 dominated through superior pattern selection (streaming generators, `execute_values`, Secrets Manager). With a tight quality-constraints SP ("testable, performant, modularised"), smaller/cheaper models closed the gap significantly — kimi-k2.5 at ~$0.001/call matched Sonnet at ~$0.13/call on overall score.

### 2. The correct MSK partial failure pattern separated the top models

Most models used `bisect_on_function_error` (blunt — retries half the batch) or silently skipped bad records (loses them). GPT-5.4 and Claude Opus/Sonnet correctly returned `{"batchItemFailures": [...]}` with exact offsets, using `report_batch_item_failures=True` in CDK. This is the production-grade pattern and a clear signal of senior AWS knowledge.

### 3. Testability was the most discriminating criterion

The SP explicitly demanded testable code. Scores ranged from 2–10 within the same run. The top models (gpt-5.4, claude-opus-4.6, claude-sonnet-4.6) all provided runnable pytest suites with meaningful fixtures and edge-case coverage. Most mid-tier models produced no tests or non-functional stubs.

### 4. DeepSeek V3.2 is exceptional value at score 8

Consistently scoring 8/10 at ~$0.003–0.005 per call — roughly 25–40× cheaper than Claude Sonnet for 1 point less on overall score. Strong on breadth and infrastructure coverage.

### 5. Thinking models need token room

kimi-k2-thinking and qwen3-vl-235b-a22b-thinking scored 1/10 in runs with max_tokens=1024 — they emit chain-of-thought before producing output. These models need at least 16k output tokens to function. All valid results used 16k–32k max_tokens.

### 6. Format constraints are the only reliable way to shape output

In a separate system-prompt comparison run (5 SP variants, same model, same prompt), only the bullet-point constraint SP ("Always respond in bullet points. Keep responses under 100 words.") reliably changed output structure. Role-only SPs ("You are a helpful assistant", "You are an expert technical writer") had negligible effect on a capable model given a clear task.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| openai | OpenAI-compatible SDK pointing at Kilo AI gateway |
| streamlit | Web UI |
| plotly | Interactive charts |
| pandas | Metrics tables |
| pydantic | Result and variant models |
| sqlite3 | Local persistence |
| python-dotenv | `.env` loading |

---

## Cost

Total spend across all runs in this benchmark series: **~$2.20**

Individual call costs ranged from $0.00003 (kimi-k2.5, short response) to $0.157 (minimax-m2.7, 12k token response).
