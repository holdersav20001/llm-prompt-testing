# Model Performance Report — AWS Lambda Code Generation
**Date:** 2026-03-28
**Task domain:** AWS Lambda functions (Kafka → S3, S3 → PostgreSQL)
**Reviewer:** OpenAI o4-mini (primary), Claude Sonnet 4.6 / Opus 4.6 (secondary)

---

## Executive Summary

12 runs across 4 distinct system prompts were analysed. The single most important finding is that **system prompt specificity is a stronger predictor of output quality than model size or cost**. A well-scoped SP levels the field significantly — cheap models (DeepSeek, Kimi, Qwen3-Coder) can equal or outperform expensive flagship models when given precise quality constraints.

---

## System Prompts Tested

| ID | Label | Summary |
|----|-------|---------|
| SP-A | Generic one-liners | `""` / `"You are a helpful assistant."` / `"expert technical writer"` / bullet-points / Alex persona |
| SP-B | Senior Developer (generic) | Long Claude skill prompt: pragmatic engineering mindset, trade-offs, code quality — no domain specifics |
| SP-C | Senior AWS Developer (plain) | AWS-specific role, clean/secure/cost-aware code, infrastructure; no explicit quality constraints |
| SP-D | Senior AWS Developer + quality constraints | SP-C plus: *"code must be testable, performant, modularised, not big chunk of code, easy to understand"* |

SP-D is the most tested configuration with the most reviewed data and forms the bulk of this analysis.

---

## SP-A: Generic One-Liners (Explanatory Task)

**Run:** 0d764b53 | **Model:** claude-haiku-4.5 only | **Prompt:** "Explain what machine learning is in simple terms"

This run compared the 5 SP-A variants on a single model — it is a **system prompt impact study**, not a model comparison. No review was generated.

| SP | Output style | Token count |
|----|-------------|------------|
| None | Structured markdown with headers | 288 |
| Helpful assistant | Near-identical to none | 270 |
| Expert technical writer | Minimal difference from above | 289 |
| Bullet points / <100 words | Followed instructions exactly — concise bullets | 127 |
| Alex persona (structured) | Three-part structure (answer / explanation / example) | 210 |

**Finding:** Format-constraint and persona SPs produce the most differentiated output. Role-only SPs ("helpful assistant", "expert writer") have negligible effect on a capable model when the task is straightforward explanation. Constrained SPs are the only reliable way to enforce output shape.

---

## SP-B: Senior Developer (Generic)

**Runs:** 014a5575, 51090286, 5c6d47dc, 691400c4, e9bbcd49
**Task:** Lambda S3→Postgres and Kafka→S3

### Model Rankings (reviewed runs)

| Run | Winner | Runner-up | Notes |
|-----|--------|-----------|-------|
| 014a5575 | **claude-sonnet-4.6** (#4) | sonnet-4.6 (#3) | haiku responses truncated at 1024; sonnet uses Secrets Manager, lru_cache, masked repr |
| 51090286 | **claude-sonnet-4.6** (#3) | haiku-4.5 (#1, #2) | Sonnet uses streaming generator + execute_values; haiku uses full in-memory load + executemany |
| e9bbcd49 | **claude-sonnet-4.6** (#2) | kimi-k2.5 (#1) | Sonnet produces modular codebase + unit tests + SAM template; Kimi is comprehensive but monolithic |

### Key Observations

- **Claude Sonnet 4.6 consistently wins under SP-B.** It selects superior architectural patterns unprompted: streaming generators (avoids OOM on large S3 files), `execute_values` bulk insert (vs `executemany`), AWS Secrets Manager (vs raw env vars).
- **Haiku 4.5 is viable for drafts but hits ceilings.** It produces solid boilerplate but defaults to simpler patterns (full file load, executemany). Good enough for a starting point, not production-ready.
- **Kimi-k2.5 is a sleeper** — comprehensive output but writes monolithic scripts that are hard to test and maintain. Scored lower despite high token count.
- **SP-B does not give models enough constraints** to enforce testability or modularity. Output quality is heavily model-dependent.

---

## SP-C: Senior AWS Developer (Plain Role, No Quality Constraints)

**Run:** de9e2c33 | **Models:** gpt-4.1-mini, deepseek-v3.2, kimi-k2.5, qwen3-235b-a22b

No numeric scores from this run (pre-scoring feature), but the narrative review was clear:

> *"Response 2 [deepseek-v3.2] is the strongest. It delivers full end-to-end CDK deployment (DLQs, alarms, VPC, IAM, concurrency limits) plus a richly instrumented Lambda handler that: batches by count/size, gzips payloads, writes partitioned S3 files, emits structured logs/X-Ray/CloudWatch, isolates poison-pills with bisect and DLQ, includes custom JSON encoding and local tests."*

| Model | Assessment |
|-------|-----------|
| **deepseek-v3.2** | Winner — most complete, full infra-as-code, poison pill handling |
| kimi-k2.5 | Strong but second; comprehensive but less infra coverage |
| qwen3-235b-a22b | Solid but not as complete |
| gpt-4.1-mini | Quickest (37s), shortest (1486 tokens) — useful scaffold, not production-ready |

**Finding:** With a domain-specific AWS SP (but no quality constraints), DeepSeek V3.2 excels at breadth — it reaches for CDK, DLQs, alarms, VPC, and observability unprompted. The AWS context in the SP activates domain knowledge that a generic SP doesn't. This is the clearest example of SP domain-specificity unlocking model capability.

---

## SP-D: Senior AWS Developer + Quality Constraints (Primary Analysis)

**Runs:** 66fe3c7f, fd6ebeb9, 324d7ddc, 05e673d9, eaafc614
**Task:** Lambda Kafka → S3 (all runs)

This SP added: *"code must be testable, performant, modularised, not big chunk of code, easy to understand and performant."*

### Scored Results

#### Run 66fe3c7f — 4 models, temp=0.8

| # | Model | Layout | Test | Perf | Exc. Mgmt | Read | Compl | Overall |
|---|-------|--------|------|------|-----------|------|-------|---------|
| 3 | **kimi-k2.5** | 9 | **10** | 9 | 9 | 9 | **10** | **9** |
| 1 | gpt-4.1-mini | 8 | 5 | 8 | 7 | 8 | 9 | 8 |
| 2 | deepseek-v3.2 | 8 | 7 | 8 | 7 | 8 | 8 | 8 |
| 4 | qwen3-235b-a22b | 7 | 4 | 7 | 6 | 8 | 6 | 6 |

Reviewer: *"Response 3 [kimi-k2.5] cleanly separates concerns into handler, processor, and I/O modules, includes pytest coverage, and handles partial failures robustly."*

#### Run 324d7ddc — 8 models × 2, temp=0.0 ⚠️ max_tokens=1024 (TRUNCATED)

All responses hit the 1024-token limit — completeness scores are artificially low (4–6). Results are comparative only; absolute scores are not meaningful.

| # | Model | Overall | Note |
|---|-------|---------|------|
| 6 | nemotron-3-super-120b-a12b | **8** | Best despite truncation |
| 14 | deepseek-v3.2 | **8** | Consistent across both runs |
| 5 | nemotron-3-super-120b-a12b | **8** | |
| 10 | glm-5-turbo | **8** | |
| 12-13 | gpt-4-turbo | **8** | |
| 1-2 | claude-haiku-4.5 | 7 | |
| 9 | kimi-k2-thinking | **1** | Only output reasoning preamble — never reached the code |

> **Critical finding:** Thinking/reasoning models (kimi-k2-thinking) require substantially more than 1024 tokens — they spend internal budget on chain-of-thought before producing output. They score 1/10 at low token limits.

#### Run 05e673d9 — 5 models, temp=0.8, full context

| # | Model | Layout | Test | Perf | Exc. Mgmt | Read | Compl | Overall |
|---|-------|--------|------|------|-----------|------|-------|---------|
| 5 | **minimax-m2.7** | 9 | **10** | **10** | 9 | 9 | **10** | **10** |
| 1 | qwen3-coder-30b-a3b | 9 | 8 | 8 | 7 | 9 | 9 | **9** |
| 3 | glm-5 | 9 | 9 | 9 | 8 | 9 | 9 | **9** |
| 4 | qwen3-vl-235b-a22b-thinking | 7 | 8 | 8 | 8 | 7 | 9 | 8 |
| 2 | nemotron-3-super-120b-a12b | 8 | 7 | 6 | **9** | 8 | 8 | 8 |

Reviewer: *"Response 5 [minimax-m2.7] provides a fully modular codebase with separate config, models, metrics, I/O, clear unit tests, CDK infra, and high-performance features (async, batching, compression)."*

#### Run eaafc614 — 2 models, temp=0.8, full context

| # | Model | Layout | Test | Perf | Exc. Mgmt | Read | Compl | Overall |
|---|-------|--------|------|------|-----------|------|-------|---------|
| 1 | **claude-sonnet-4.6** | 9 | 9 | 8 | 9 | 9 | 9 | **9** |
| 2 | minimax-m2.7 | 8 | 6 | 7 | 6 | 7 | 8 | 7 |

Reviewer (claude-sonnet-4.6 self-review with caution): *"Response 2's tests have significant issues — mocker not declared as fixture parameter (a bug), auto-flush behaviour not actually implemented, parametrize usage wrong. Response 2 also computes json.dumps twice per record and swallows exceptions silently."*

> **Note:** minimax-m2.7 scored 10/10 in run 05e673d9 but only 7/10 in eaafc614. This suggests **high variance** — the model is capable of excellent output but inconsistent.

---

## Consolidated Model Rankings (SP-D, Code Generation)

Based on all scored SP-D runs:

| Model | Best Score | Typical Score | Latency | Cost (7k tokens) | Verdict |
|-------|-----------|---------------|---------|-----------------|---------|
| **kimi-k2.5** | 9 | 9 | ~120–165s | ~$0.001 | Best testability, exceptional value — but slow |
| **qwen3-coder-30b** | 9 | 9 | ~10–27s | ~$0.03 | Best speed/quality ratio for SP-D tasks |
| **glm-5** | 9 | 9 | ~120s | ~$0.07 | Solid and consistent |
| **claude-sonnet-4.6** | 9 | 9 | ~68–122s | ~$0.09–0.13 | Reliable, elegant patterns, most expensive |
| **deepseek-v3.2** | 8 | 8 | ~70–120s | ~$0.003–0.005 | Best value at score 8; excels at breadth/infra |
| **minimax-m2.7** | 10 | 7–10 | ~160–210s | ~$0.12–0.16 | Peak ceiling but high variance |
| gpt-4.1-mini | 8 | 8 | ~18–37s | ~$0.002 | Fastest, least complete on testability |
| nemotron-3-super-120b | 8 | 8 | ~10–17s | ~$0.05 | Fast, strong exception management |
| qwen3-235b-a22b | 6–8 | 7 | ~130–160s | ~$0.02–0.05 | Inconsistent; verbose without the coder variant |
| glm-4.7 / glm-5-turbo | 7–8 | 7 | ~50–70s | ~$0.07 | Decent but outclassed within the z-ai family by glm-5 |
| kimi-k2-thinking | 1–8 | varies | ~235–360s | ~$0.001 | Requires 16k+ tokens; scores 1/10 at 1024 |
| gpt-4-turbo | 8 | 8 | ~40s | ~$0.02 | Solid but not better than newer smaller models |

---

## Key Findings

### 1. SP specificity is the primary quality lever
Under SP-B (generic), sonnet-4.6 dominates through superior pattern selection (streaming, execute_values, Secrets Manager). Under SP-D (explicit quality constraints), cheap models equalise — kimi-k2.5 at $0.001 matches sonnet-4.6 at $0.13 on overall score.

### 2. Testability is where models diverge most
The SP explicitly demands testability. Scores on this criterion span 4–10 within a single run. kimi-k2.5 and minimax-m2.7 score 10/10; gpt-4.1-mini scores 5/10. **If testability matters, kimi-k2.5 or qwen3-coder are the right choices.**

### 3. Thinking models need room to think
kimi-k2-thinking, qwen3-vl-235b-a22b-thinking: Both need 16k–32k output tokens to express their reasoning before producing code. At 1024 tokens, kimi-k2-thinking scored 1/10 (only emitted chain-of-thought). Always set max_tokens ≥ 16384 for these models.

### 4. DeepSeek V3.2 punches far above its cost
Consistently 8/10 across SP-C and SP-D runs. At $0.003–0.005 per call it is ~25–40× cheaper than sonnet-4.6 for 1 point less on overall score. Strong on breadth and infra; slightly weaker on testability.

### 5. minimax-m2.7 shows high ceiling but inconsistency
10/10 in one run (async, batching, CDK, full tests), 7/10 in another (test bugs, swallowed exceptions). Not suitable as a primary model until consistency improves.

### 6. Token truncation invalidates code generation results
Any run with max_tokens < 4096 for code tasks should be treated as invalid. The 324d7ddc run (1024 tokens) shows this clearly — every model's completeness score is 4–6 regardless of actual capability.

---

## Recommendations

| Goal | Recommended SP | Recommended Model |
|------|---------------|-------------------|
| Production-grade code, best quality | SP-D | claude-sonnet-4.6 or kimi-k2.5 |
| Best testability | SP-D | kimi-k2.5 (score 10) or qwen3-coder-30b |
| Best speed + quality | SP-D | qwen3-coder-30b (10s, score 9) or gpt-4.1-mini (18s, score 8) |
| Best cost efficiency | SP-D | deepseek-v3.2 (score 8, $0.003) or kimi-k2.5 (score 9, $0.001) |
| Broadest AWS infra coverage | SP-C | deepseek-v3.2 |
| Batch experiments / model survey | SP-D | Run qwen3-coder-30b as baseline (fast + cheap + consistent) |

**Always:** set max_tokens ≥ 16384 for code tasks. Use temperature 0.0 for deterministic comparison; 0.8 for exploring the model's range.
