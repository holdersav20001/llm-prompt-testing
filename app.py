import uuid
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from src.client import ClaudeClient
from src.storage import SQLiteStorage
from src.config import ExperimentConfig, Variant
from src.experiments.matrix import MatrixExperiment
from src import visualisation

st.set_page_config(page_title="Claude API Explorer", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_available_models() -> list[str]:
    """Fetch live model list from Kilo AI gateway, sorted alphabetically."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("KILO_API_KEY", ""),
            base_url="https://api.kilo.ai/api/gateway",
        )
        models = client.models.list()
        return sorted(m.id for m in models.data)
    except Exception:
        # Fallback: curated list if API is unreachable
        return [
            "anthropic/claude-haiku-4.5",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-opus-4.6",
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/o4-mini",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-r1",
            "x-ai/grok-4",
        ]


AVAILABLE_MODELS = _fetch_available_models()

REVIEW_MODELS = [
    "openai/o4-mini",
    "openai/o3",
    "openai/o3-mini",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-3.7-sonnet:thinking",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-r1",
    "x-ai/grok-4",
]

CHART_FN = {
    "matrix": visualisation.plot_matrix_summary,
    "system_prompts": visualisation.plot_system_prompts,
    "model_compare": visualisation.plot_model_comparison,
    "token_tracking": visualisation.plot_token_usage,
    "streaming": visualisation.plot_streaming,
    "temperature_sweep": visualisation.plot_temperature_sweep,
}


REVIEW_SYSTEM_PROMPT = """\
You are a senior software engineer reviewing AI-generated responses.
Score each response 1–10 (1=poor, 10=excellent) on these six criteria:
- Layout: structure, formatting, organisation
- Testability: ease of testing, separation of concerns
- Performance: efficiency, complexity, resource usage
- Exception Management: error handling, edge cases, robustness
- Readability: naming, comments, clarity
- Completeness: fully addresses the requirement

Set "overall" to the SUM of all six scores (max 60).

Return your assessment in EXACTLY this format — no deviations:

SCORES:
```json
{"responses": [{"n": 1, "layout": 0, "testability": 0, "performance": 0, "exception_management": 0, "readability": 0, "completeness": 0, "overall": 0}]}
```

ASSESSMENT:
[Narrative: which is best, why, and key differences. Be concise.]\
"""


import re, json as _json

def parse_review_scores(review_text: str) -> dict[int, dict]:
    """Extract per-response scores from a review. Returns {n: {criteria: score}}."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", review_text, re.DOTALL)
    if not match:
        return {}
    try:
        data = _json.loads(match.group(1))
        return {r["n"]: r for r in data.get("responses", [])}
    except Exception:
        return {}


def build_review_prompt(results: list) -> str:
    parts = ["Review the following AI responses and score each one.\n\n"]
    for i, r in enumerate(results, 1):
        parts.append(
            f"--- Response {i} ---\n"
            f"Model: {r.model.split('/')[-1]}\n"
            f"System prompt: {r.params.get('system_prompt', 'none')}\n"
            f"Prompt: {r.params.get('prompt', r.prompt)}\n"
            f"Temperature: {r.temperature}\n\n"
            f"{r.response_text}\n\n"
        )
    return "".join(parts)


def fmt_cost(cost: float) -> str:
    if cost == 0:
        return "$0.00"
    if cost < 0.0001:
        return f"${cost:.2e}"
    if cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.4f}"


# ── Resources ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return ClaudeClient()

@st.cache_resource
def get_storage():
    return SQLiteStorage()

client = get_client()
storage = get_storage()

if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0

# ── Page header ───────────────────────────────────────────────────────────────
st.title("Claude API Explorer")
spend = st.session_state.session_cost
if spend >= 1.00:
    st.warning(f"Session spend: {fmt_cost(spend)} — over $1.00 threshold")
else:
    st.caption(f"Session spend: {fmt_cost(spend)}")

tab_run, tab_results, tab_compare = st.tabs(["Configure & Run", "Results", "Compare"])


# ── Variant manager helper ────────────────────────────────────────────────────
def render_variants(variant_type: str, prefix: str) -> list[dict]:
    """
    Renders a managed list of variants with checkboxes, edit and delete.
    Returns list of selected variants as dicts (name, content).
    """
    variants = storage.get_variants(variant_type)

    sel_key = f"{prefix}_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = {v.id for v in variants}

    editing_key = f"{prefix}_editing"
    if editing_key not in st.session_state:
        st.session_state[editing_key] = None

    adding_key = f"{prefix}_adding"
    if adding_key not in st.session_state:
        st.session_state[adding_key] = False

    # Variant rows
    for v in variants:
        # Paused variants are forced out of the selected set
        if v.paused:
            st.session_state[sel_key].discard(v.id)

        is_sel = v.id in st.session_state[sel_key]
        c_chk, c_name, c_pause, c_edit, c_del = st.columns([0.07, 0.50, 0.16, 0.14, 0.13])

        # Checkbox disabled when paused
        if not v.paused:
            new_sel = c_chk.checkbox(" ", value=is_sel, key=f"{prefix}_chk_{v.id}", label_visibility="collapsed")
            if new_sel:
                st.session_state[sel_key].add(v.id)
            else:
                st.session_state[sel_key].discard(v.id)
        else:
            c_chk.checkbox(" ", value=False, key=f"{prefix}_chk_{v.id}", disabled=True, label_visibility="collapsed")

        name_display = f"~~{v.name}~~" if v.paused else f"**{v.name}**"
        c_name.markdown(name_display)
        preview = v.content[:50] + "…" if len(v.content) > 50 else (v.content or "*(empty)*")
        caption = f"*paused* — {preview}" if v.paused else preview
        c_name.caption(caption)

        pause_label = "Resume" if v.paused else "Pause"
        if c_pause.button(pause_label, key=f"{prefix}_pause_{v.id}", use_container_width=True):
            storage.set_variant_paused(v.id, not v.paused)
            st.rerun()

        if c_edit.button("Edit", key=f"{prefix}_edit_{v.id}", use_container_width=True):
            st.session_state[editing_key] = v.id
            st.session_state[adding_key] = False
            st.rerun()

        if c_del.button("Del", key=f"{prefix}_del_{v.id}", use_container_width=True):
            storage.delete_variant(v.id)
            st.session_state[sel_key].discard(v.id)
            st.rerun()

    # Edit form
    editing_id = st.session_state[editing_key]
    if editing_id:
        edit_v = next((v for v in variants if v.id == editing_id), None)
        if edit_v:
            with st.form(key=f"{prefix}_edit_form"):
                st.caption(f"Editing: {edit_v.name}")
                new_name = st.text_input("Name", value=edit_v.name)
                new_content = st.text_area("Content", value=edit_v.content, height=90)
                sc1, sc2 = st.columns(2)
                if sc1.form_submit_button("Save", use_container_width=True):
                    storage.update_variant(editing_id, new_name, new_content)
                    st.session_state[editing_key] = None
                    st.rerun()
                if sc2.form_submit_button("Cancel", use_container_width=True):
                    st.session_state[editing_key] = None
                    st.rerun()

    # Add form
    if st.session_state[adding_key]:
        with st.form(key=f"{prefix}_add_form"):
            st.caption("New variant")
            new_name = st.text_input("Name")
            new_content = st.text_area("Content", height=90)
            ac1, ac2 = st.columns(2)
            if ac1.form_submit_button("Add", use_container_width=True):
                if new_name.strip():
                    v = Variant(type=variant_type, name=new_name.strip(), content=new_content)
                    storage.save_variant(v)
                    st.session_state[sel_key].add(v.id)
                st.session_state[adding_key] = False
                st.rerun()
            if ac2.form_submit_button("Cancel", use_container_width=True):
                st.session_state[adding_key] = False
                st.rerun()
    else:
        if st.button("+ Add", key=f"{prefix}_add_btn"):
            st.session_state[adding_key] = True
            st.session_state[editing_key] = None
            st.rerun()

    # Return selected as dicts
    selected_ids = st.session_state[sel_key]
    return [{"name": v.name, "content": v.content} for v in variants if v.id in selected_ids]


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Configure & Run
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_right:
        sp_col, pv_col = st.columns(2, gap="medium")

        with sp_col:
            st.subheader("System Prompt Variants")
            selected_sp = render_variants("system_prompt", "sp")

        with pv_col:
            st.subheader("Prompt Variants")
            selected_pv = render_variants("prompt_length", "pv")

    with col_left:
        st.subheader("Models")
        selected_models = st.multiselect(
            "Select models",
            options=AVAILABLE_MODELS,
            default=["anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-4.6"],
            format_func=lambda m: m.split("/")[-1],
            label_visibility="collapsed",
        )

        st.subheader("Temperature")
        temps: list[float] = []
        temp_options = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        t_cols = st.columns(len(temp_options))
        for i, t in enumerate(temp_options):
            default = (t == 0.8)
            if t_cols[i].checkbox(str(t), value=default, key=f"t{i}"):
                temps.append(t)

        streaming = st.checkbox("Measure streaming / TTFT", key="streaming")

        st.subheader("Max Tokens")
        max_tokens = st.selectbox(
            "Max output tokens per call",
            options=list(range(1024, 65537, 1024)),
            index=31,
            key="max_tokens",
            label_visibility="collapsed",
        )

        st.divider()

        n_sp = max(len(selected_sp), 1)
        n_pv = max(len(selected_pv), 1)
        n_t  = max(len(temps), 1)
        n_m  = max(len(selected_models), 1)
        n_calls = n_m * n_sp * n_pv * n_t * (2 if streaming else 1)

        mc1, mc2 = st.columns(2)
        mc1.metric("API Calls", n_calls)
        mc2.metric("Session Spend", fmt_cost(st.session_state.session_cost))

        # Validation
        errors = []
        if not selected_models:
            errors.append("Select at least one model.")
        if not temps:
            errors.append("Select at least one temperature.")
        if not selected_sp:
            errors.append("Select at least one system prompt variant.")
        if not selected_pv:
            errors.append("Select at least one prompt variant.")

        for e in errors:
            st.warning(e)

        run_disabled = bool(errors)
        if st.button("Run", type="primary", use_container_width=True, disabled=run_disabled):
            run_id = str(uuid.uuid4())
            config = ExperimentConfig(name="matrix", prompt="", model=selected_models[0], max_tokens=max_tokens)

            progress_table = st.empty()

            def update_progress(rows):
                df = pd.DataFrame(rows)
                progress_table.dataframe(df, use_container_width=True, hide_index=True)

            with st.spinner(f"Running {n_calls} API calls…"):
                try:
                    exp = MatrixExperiment(
                        client=client,
                        storage=storage,
                        config=config,
                        models=selected_models,
                        system_prompt_variants=selected_sp,
                        prompt_variants=selected_pv,
                        temperatures=temps,
                        streaming=streaming,
                        run_id=run_id,
                    )
                    results = exp.run(on_update=update_progress)
                    total_cost = sum(r.cost_usd for r in results)
                    st.session_state.session_cost += total_cost
                    st.success(
                        f"Done — {len(results)} calls, {fmt_cost(total_cost)} spent.  "
                        f"Run `{run_id[:8]}`"
                    )
                    st.plotly_chart(
                        visualisation.plot_matrix_summary(results),
                        use_container_width=True,
                        key=f"inline_{run_id}",
                    )
                except Exception as e:
                    st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    runs = storage.get_runs()

    if not runs:
        st.info("No runs yet. Run an experiment first.")
    else:
        st.subheader(f"Completed Runs  ({len(runs)})")

        for run in runs:
            run_id = run["run_id"]
            run_results = storage.get_by_run_id(run_id)
            if not run_results:
                continue

            # Derive parameters from stored results
            models_used = sorted({r.model.split("/")[-1] for r in run_results})
            sp_names = sorted({r.params.get("system_prompt", "?") for r in run_results})
            pv_names = sorted({r.params.get("prompt", "?") for r in run_results})
            temps = sorted({r.temperature for r in run_results if r.temperature is not None})
            has_streaming = any(r.params.get("mode") == "stream" for r in run_results)

            total_input = sum(r.input_tokens for r in run_results)
            total_output = sum(r.output_tokens for r in run_results)
            total_cost = sum(r.cost_usd for r in run_results)
            latencies = [r.latency_ms for r in run_results]
            avg_lat = int(sum(latencies) / len(latencies))
            min_lat = min(latencies)
            max_lat = max(latencies)

            with st.container(border=True):
                # Header row: ID · timestamp · models
                h1, h2 = st.columns([3, 1])
                h1.markdown(
                    f"**`{run_id[:8]}`**  ·  {run['started_at'][:16].replace('T', ' ')}  ·  "
                    + "  ".join(f"`{m}`" for m in models_used)
                )
                h2.markdown(f"**{fmt_cost(total_cost)}**  ·  {len(run_results)} calls")

                # Parameters row
                param_str = (
                    f"System prompts: {', '.join(sp_names)}  |  "
                    f"Prompts: {', '.join(pv_names)}  |  "
                    f"Temps: {', '.join(str(t) for t in temps)}"
                )
                if has_streaming:
                    param_str += "  |  + streaming"
                st.caption(param_str)

                # Metrics row
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Tokens In", f"{total_input:,}")
                mc2.metric("Tokens Out", f"{total_output:,}")
                mc3.metric("Total Cost", fmt_cost(total_cost))
                mc4.metric("Avg Latency", f"{avg_lat} ms")
                mc5.metric("Min / Max", f"{min_lat} / {max_lat} ms")

                # AI review
                rb1, rb2, rb3 = st.columns([2, 1, 3])
                review_model = rb1.selectbox(
                    "Review model",
                    REVIEW_MODELS,
                    format_func=lambda m: m.split("/")[-1],
                    key=f"review_model_{run_id}",
                    label_visibility="collapsed",
                )
                if rb2.button("Review", key=f"review_btn_{run_id}", use_container_width=True):
                    batch_results = [r for r in run_results if r.params.get("mode", "batch") == "batch"]
                    with st.spinner(f"Reviewing with {review_model.split('/')[-1]}…"):
                        try:
                            result = client.call(
                                model=review_model,
                                prompt=build_review_prompt(batch_results),
                                system_prompt=REVIEW_SYSTEM_PROMPT,
                                max_tokens=32000,
                            )
                            storage.save_review(run_id, review_model, result["response_text"])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Review error: {e}")

                batch_only = [r for r in run_results if r.params.get("mode", "batch") == "batch"]
                stream_only = [r for r in run_results if r.params.get("mode") == "stream"]

                all_reviews = storage.get_reviews(run_id)

                def _render_review(rev, expanded=True):
                    scores = parse_review_scores(rev["review_text"])
                    if scores:
                        score_rows = []
                        for n, s in sorted(scores.items()):
                            model_name = batch_only[n - 1].model.split("/")[-1] if n - 1 < len(batch_only) else "?"
                            sp_name = batch_only[n - 1].params.get("system_prompt", "") if n - 1 < len(batch_only) else ""
                            score_rows.append({
                                "#": n,
                                "Model": model_name,
                                "SP": sp_name,
                                "Layout": s.get("layout", ""),
                                "Testability": s.get("testability", ""),
                                "Performance": s.get("performance", ""),
                                "Exc. Mgmt": s.get("exception_management", ""),
                                "Readability": s.get("readability", ""),
                                "Completeness": s.get("completeness", ""),
                                "Overall": s.get("overall", ""),
                            })
                        st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
                    narrative = re.sub(r"SCORES:\s*```json.*?```", "", rev["review_text"], flags=re.DOTALL).replace("ASSESSMENT:", "").strip()
                    st.markdown(narrative)

                if all_reviews:
                    latest = all_reviews[0]
                    with st.container(border=True):
                        st.caption(f"**{latest['model'].split('/')[-1]}**  ·  {latest['created_at'][:16].replace('T', ' ')}")
                        _render_review(latest)
                    if len(all_reviews) > 1:
                        with st.expander(f"Previous reviews ({len(all_reviews) - 1})"):
                            for rev in all_reviews[1:]:
                                st.caption(f"**{rev['model'].split('/')[-1]}**  ·  {rev['created_at'][:16].replace('T', ' ')}")
                                _render_review(rev)
                                st.divider()

                ec1, ec2 = st.columns(2)

                with ec1.expander("View metrics"):
                    has_ttft = any(r.ttft_ms for r in run_results)
                    # Get scores from latest review if available
                    latest_reviews = storage.get_reviews(run_id)
                    scores = parse_review_scores(latest_reviews[0]["review_text"]) if latest_reviews else {}
                    rows = []
                    for i, r in enumerate(batch_only, 1):
                        row = {
                            "#": i,
                            "Model": r.model.split("/")[-1],
                            "System Prompt": r.params.get("system_prompt", ""),
                            "Prompt": r.prompt,
                            "Temp": r.temperature,
                            "Tokens In": r.input_tokens,
                            "Tokens Out": r.output_tokens,
                            "Cost": fmt_cost(r.cost_usd),
                            "Latency (ms)": r.latency_ms,
                        }
                        if has_ttft:
                            row["TTFT (ms)"] = r.ttft_ms or ""
                        if scores.get(i):
                            s = scores[i]
                            row["Layout"] = s.get("layout", "")
                            row["Testability"] = s.get("testability", "")
                            row["Performance"] = s.get("performance", "")
                            row["Exc. Mgmt"] = s.get("exception_management", "")
                            row["Readability"] = s.get("readability", "")
                            row["Completeness"] = s.get("completeness", "")
                            row["Overall"] = s.get("overall", "")
                        rows.append(row)
                    if stream_only:
                        for r in stream_only:
                            row = {
                                "#": "S",
                                "Model": r.model.split("/")[-1],
                                "System Prompt": r.params.get("system_prompt", ""),
                                "Prompt": r.prompt,
                                "Temp": r.temperature,
                                "Tokens In": r.input_tokens,
                                "Tokens Out": r.output_tokens,
                                "Cost": fmt_cost(r.cost_usd),
                                "Latency (ms)": r.latency_ms,
                                "TTFT (ms)": r.ttft_ms or "",
                            }
                            rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                with ec2.expander("View results"):
                    for i, r in enumerate(batch_only, 1):
                        param_parts = [f"{k}={v}" for k, v in r.params.items() if k not in ("model", "mode")]
                        call_header = f"**#{i}  {r.model.split('/')[-1]}**"
                        if param_parts:
                            call_header += "  —  " + "  |  ".join(param_parts)
                        with st.container(border=True):
                            st.markdown(call_header)
                            st.text(r.response_text)
                    for r in stream_only:
                        param_parts = [f"{k}={v}" for k, v in r.params.items() if k not in ("model", "mode")]
                        call_header = f"**#S  {r.model.split('/')[-1]}  (stream)**"
                        if param_parts:
                            call_header += "  —  " + "  |  ".join(param_parts)
                        with st.container(border=True):
                            st.markdown(call_header)
                            st.text(r.response_text)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Compare
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    all_results = storage.get_all()
    batch_all = [r for r in all_results if r.params.get("mode", "batch") == "batch" and r.response_text]

    if not batch_all:
        st.info("No results yet. Run an experiment first.")
    else:
        st.subheader("Select responses to compare")
        st.caption("Click rows to select, then choose a model and hit Compare.")

        cmp_rows = [
            {
                "id": r.id,
                "Run": r.run_id[:8] if r.run_id else "—",
                "Model": r.model.split("/")[-1],
                "System Prompt": r.params.get("system_prompt", ""),
                "Prompt": r.prompt,
                "Temp": r.temperature,
                "Tokens Out": r.output_tokens,
                "Preview": r.response_text[:100] + "…" if len(r.response_text) > 100 else r.response_text,
            }
            for r in batch_all
        ]

        cmp_df = pd.DataFrame(cmp_rows)
        event = st.dataframe(
            cmp_df.drop(columns=["id"]),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
        )

        selected_indices = event.selection.rows
        selected_ids = {cmp_rows[i]["id"] for i in selected_indices}
        selected_results = [r for r in batch_all if r.id in selected_ids]

        if selected_results:
            st.caption(f"{len(selected_results)} response(s) selected")
            cm1, cm2, _ = st.columns([2, 1, 3])
            compare_model = cm1.selectbox(
                "Compare with",
                REVIEW_MODELS,
                format_func=lambda m: m.split("/")[-1],
                key="compare_model",
                label_visibility="collapsed",
            )
            if cm2.button("Compare", key="compare_btn", use_container_width=True):
                with st.spinner(f"Comparing with {compare_model.split('/')[-1]}…"):
                    try:
                        result = client.call(
                            model=compare_model,
                            prompt=build_review_prompt(selected_results),
                            system_prompt=REVIEW_SYSTEM_PROMPT,
                            max_tokens=32000,
                        )
                        storage.save_comparison(
                            [r.id for r in selected_results],
                            compare_model,
                            result["response_text"],
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Compare error: {e}")

        # Saved comparisons
        comparisons = storage.get_comparisons()
        if comparisons:
            st.divider()
            st.subheader(f"Saved Comparisons  ({len(comparisons)})")
            for c in comparisons:
                with st.container(border=True):
                    st.caption(
                        f"**{c['model'].split('/')[-1]}**  ·  "
                        f"{c['created_at'][:16].replace('T', ' ')}  ·  "
                        f"{len(c['result_ids'])} responses"
                    )
                    scores = parse_review_scores(c["comparison_text"])
                    if scores:
                        result_map = {r.id: r for r in batch_all}
                        score_rows = []
                        for n, s in sorted(scores.items()):
                            rid = c["result_ids"][n - 1] if n - 1 < len(c["result_ids"]) else None
                            model_name = result_map[rid].model.split("/")[-1] if rid and rid in result_map else "?"
                            score_rows.append({
                                "#": n,
                                "Model": model_name,
                                "Layout": s.get("layout", ""),
                                "Testability": s.get("testability", ""),
                                "Performance": s.get("performance", ""),
                                "Exc. Mgmt": s.get("exception_management", ""),
                                "Readability": s.get("readability", ""),
                                "Completeness": s.get("completeness", ""),
                                "Overall": s.get("overall", ""),
                            })
                        st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
                    narrative = re.sub(r"SCORES:\s*```json.*?```", "", c["comparison_text"], flags=re.DOTALL).replace("ASSESSMENT:", "").strip()
                    st.markdown(narrative)
