from pathlib import Path

from ui_next.theme import THEME_CSS, compact_header_html, transition_overlay_html


def test_watercast_header_is_independent_and_escapes_dynamic_text() -> None:
    rendered = compact_header_html(
        '<script>alert("x")</script>',
        "API 정상 & 모델",
        connected=True,
    )
    assert "WATER" in rendered and "CAST" in rendered
    assert "AEGIS" not in rendered.upper()
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "API 정상 &amp; 모델" in rendered
    assert "SATELLITE ANALYTICS CONSOLE" in rendered
    assert "wc-topbar__status--online" in rendered


def test_transition_is_short_pointer_safe_and_accessible() -> None:
    rendered = transition_overlay_html("결과 <화면>")
    assert "wc-transition" in rendered
    assert "pointer-events: none" in THEME_CSS
    assert ".62s" in THEME_CSS
    assert "prefers-reduced-motion" in THEME_CSS
    assert "결과 &lt;화면&gt;" in rendered


def test_streamlit_161_glide_dataframe_has_dark_canvas_and_dom_overlays() -> None:
    required_selectors = (
        '[data-testid="stDataFrameResizable"]',
        '.stDataFrameGlideDataEditor',
        'canvas[data-testid="data-grid-canvas"]',
        '[data-testid="stElementToolbarButtonContainer"]',
        '[data-testid="stBaseButton-elementToolbar"]',
        '[data-testid="stDataFrameColumnVisibilityMenu"]',
        '[data-testid="stDataFrameColumnMenu"]',
        '[data-testid="stDataFrameColumnFormattingMenu"]',
        '[data-testid="stDataFrameStatisticsMenu"]',
        '[data-testid="stDataFrameButtonActionMenu"]',
        '[data-testid="stFullScreenFrame"]:fullscreen',
    )
    for selector in required_selectors:
        assert selector in THEME_CSS

    required_glide_variables = (
        "--gdg-bg-cell: #0b1324",
        "--gdg-bg-header: #111c31",
        "--gdg-text-dark: #e2e8f0",
        "--gdg-border-color: #1e293b",
        "--gdg-accent-color: #22d3ee",
        "--gdg-link-color: #38bdf8",
    )
    for declaration in required_glide_variables:
        assert declaration in THEME_CSS

    # Glide paints visible cells on canvas, so th/td rules alone are not enough.
    # Native Streamlit dark theme must drive the bitmap; inversion would also
    # invert an already-dark grid and can hide its cell text.
    config = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"
    config_text = config.read_text(encoding="utf-8")
    assert 'base = "dark"' in config_text
    assert 'secondaryBackgroundColor = "#0b1324"' in config_text
    assert "filter: invert" not in THEME_CSS
    assert ".gdg-input" in THEME_CSS
    assert ".gdg-search-bar-inner" in THEME_CSS


def test_streamlit_chrome_does_not_reserve_a_blank_top_bar() -> None:
    assert '[data-testid="stHeader"]' in THEME_CSS
    assert "height: 0 !important" in THEME_CSS
    assert '[data-testid="stToolbar"]' in THEME_CSS
    assert "display: none !important" in THEME_CSS
    assert '[data-testid="stExpandSidebarButton"]' in THEME_CSS
    assert '[data-testid="stAppDeployButton"]' in THEME_CSS
    assert '[data-testid="stMainMenu"]' in THEME_CSS
    assert '[data-testid="stSidebarHeader"]' in THEME_CSS


def test_select_date_and_guide_helpers_do_not_fall_back_to_white() -> None:
    assert '[data-baseweb="calendar"]' in THEME_CSS
    assert '[data-baseweb="select"] > div' in THEME_CSS
    assert '[data-st-baseweb-layer-host="true"] [data-baseweb="popover"]' in THEME_CSS
    assert "background-color: #0b1324 !important" in THEME_CSS

    for helper in (
        ".wc-guide-hero",
        ".wc-guide-card",
        ".wc-term",
        ".wc-explain",
        ".wc-present-note",
        ".wc-step-number",
        ".wc-legend-row",
    ):
        assert helper in THEME_CSS
