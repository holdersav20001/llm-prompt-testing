"""
Playwright UI tests for the Claude API Explorer Streamlit app.
Requires the app to be running on http://localhost:8501.
Run with: pytest tests/test_ui.py --headed   (or headless by default)
"""
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8501"
TIMEOUT = 15_000  # ms


@pytest.fixture(scope="session")
def browser_context_args():
    return {"base_url": BASE_URL}


# ── Helpers ───────────────────────────────────────────────────────────────────

def wait_for_streamlit(page: Page):
    """Wait until Streamlit finishes rendering (spinner gone, no skeleton loaders)."""
    page.wait_for_load_state("networkidle", timeout=TIMEOUT)
    # Wait for the Streamlit app root to appear
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=TIMEOUT)


def get_tab(page: Page, label: str):
    return page.get_by_role("tab", name=label)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_app_loads(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    expect(page).to_have_title("Claude API Explorer")


def test_page_has_two_tabs(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    expect(get_tab(page, "Configure & Run")).to_be_visible()
    expect(get_tab(page, "Results")).to_be_visible()


def test_configure_tab_has_model_selector(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    # Multi-select for models
    expect(page.get_by_text("Select models")).to_be_visible()


def test_configure_tab_has_experiment_selector(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    expect(page.get_by_text("What to test")).to_be_visible()


def test_configure_tab_has_temperature_input(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    expect(page.get_by_text("Temperature")).to_be_visible()


def test_configure_tab_has_prompt_textarea(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    # Default prompt text should be present
    expect(page.get_by_text("Explain what machine learning is in simple terms.")).to_be_visible()


def test_configure_tab_has_run_button(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    expect(page.get_by_role("button", name="Run Experiment")).to_be_visible()


def test_temperature_sweep_hides_temperature_input(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    # Select Temperature Sweep from the experiment dropdown
    exp_select = page.locator("[data-testid='stSelectbox']").filter(has_text="What to test")
    exp_select.locator("div[data-baseweb='select']").click()
    page.get_by_role("option", name="Temperature Sweep").click()
    wait_for_streamlit(page)
    # The auto-sweep info message should appear
    expect(page.get_by_text("Sweeps 0.0, 0.1, 0.5, 1.0 automatically.")).to_be_visible()
    # Temperature number input should be hidden
    expect(page.get_by_test_id("stNumberInputField")).not_to_be_visible()


def test_temperature_input_accepts_custom_value(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    temp_input = page.get_by_test_id("stNumberInputField")
    temp_input.fill("0.7")
    temp_input.press("Tab")
    wait_for_streamlit(page)
    expect(temp_input).to_have_value("0.7")


def test_results_tab_shows_empty_state(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    get_tab(page, "Results").click()
    wait_for_streamlit(page)
    # Either shows a runs dataframe or the "No runs yet" info message
    runs_table = page.locator("[data-testid='stDataFrame']")
    empty_msg = page.get_by_text("No runs yet", exact=False)
    try:
        expect(runs_table.or_(empty_msg)).to_be_visible(timeout=8_000)
    except Exception:
        # Fall back: check each separately
        has_runs = runs_table.count() > 0
        has_empty = empty_msg.count() > 0
        assert has_runs or has_empty, "Results tab should show runs table or empty state"


def test_session_spend_displayed(page: Page):
    page.goto(BASE_URL)
    wait_for_streamlit(page)
    expect(page.get_by_text("Session spend:")).to_be_visible()


def test_experiment_descriptions_update(page: Page):
    """Switching experiment type updates the description caption."""
    page.goto(BASE_URL)
    wait_for_streamlit(page)

    descriptions = {
        "System Prompts": "Tests 5 system prompt styles",
        "Model Comparison": "Runs the same prompt across all selected models",
        "Temperature Sweep": "Sweeps",
    }

    exp_select = page.locator("[data-testid='stSelectbox']").filter(has_text="What to test")
    for exp_name, expected_text in descriptions.items():
        exp_select.locator("div[data-baseweb='select']").click()
        page.get_by_role("option", name=exp_name).click()
        wait_for_streamlit(page)
        expect(page.get_by_text(expected_text, exact=False)).to_be_visible()
