"""WATERCAST visual system for the Streamlit operator console.

The module deliberately contains no external font, image, or brand dependency.
Call :func:`inject_theme` once, immediately after ``st.set_page_config``.  The
other helpers return escaped, self-contained HTML suitable for
``st.markdown(..., unsafe_allow_html=True)``.
"""

from __future__ import annotations

from html import escape

THEME_CSS = r"""
<style>
:root {
  color-scheme: dark;
  --wc-bg: #020617;
  --wc-surface: #0f172a;
  --wc-surface-soft: rgba(15, 23, 42, .78);
  --wc-surface-raised: #111c31;
  --wc-border: #1e293b;
  --wc-border-strong: #334155;
  --wc-text: #f1f5f9;
  --wc-muted: #94a3b8;
  --wc-dim: #64748b;
  --wc-cyan: #22d3ee;
  --wc-sky: #38bdf8;
  --wc-sky-strong: #0284c7;
  --wc-green: #22c55e;
  --wc-yellow: #eab308;
  --wc-orange: #f97316;
  --wc-red: #ef4444;
  --wc-purple: #a855f7;
  --wc-radius: 8px;
  --wc-shadow: 0 16px 38px rgba(0, 0, 0, .24);
}

html, body, [class*="css"] {
  font-family: Inter, Pretendard, "Noto Sans KR", ui-sans-serif, system-ui,
    -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--wc-bg) !important;
  color: var(--wc-text) !important;
}

.stApp {
  background-image:
    radial-gradient(circle at 72% -18%, rgba(14, 165, 233, .10), transparent 32rem),
    linear-gradient(rgba(56, 189, 248, .016) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56, 189, 248, .016) 1px, transparent 1px) !important;
  background-size: auto, 32px 32px, 32px 32px !important;
}

[data-testid="stHeader"] {
  height: 0 !important;
  min-height: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

[data-testid="stHeaderActionElements"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
  display: none !important;
}

/* Keep only the sidebar opener from Streamlit's otherwise hidden chrome. */
[data-testid="stToolbar"] {
  position: fixed !important;
  inset: .65rem auto auto .7rem !important;
  z-index: 1200 !important;
  display: flex !important;
  width: 2.4rem !important;
  height: 2.4rem !important;
  overflow: visible !important;
}
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] {
  display: none !important;
}
[data-testid="stExpandSidebarButton"] {
  width: 2rem !important;
  height: 2rem !important;
  border: 1px solid var(--wc-border-strong) !important;
  border-radius: 7px !important;
  color: var(--wc-muted) !important;
  background: rgba(7, 16, 31, .94) !important;
  box-shadow: 0 8px 22px rgba(0, 0, 0, .22) !important;
}

.block-container,
[data-testid="stMainBlockContainer"] {
  width: 100%;
  max-width: 1600px;
  padding: 1rem 1rem 3rem !important;
}

h1, h2, h3, h4, h5, h6,
p, label, span, div, li {
  color: inherit;
}

h1 { font-size: 1.75rem !important; letter-spacing: -.025em; }
h2 { font-size: 1.32rem !important; letter-spacing: -.015em; }
h3 { font-size: 1.05rem !important; }

a, a:visited {
  color: var(--wc-sky) !important;
  text-decoration-color: rgba(56, 189, 248, .45) !important;
}

small, .stCaption, [data-testid="stCaptionContainer"] {
  color: var(--wc-muted) !important;
}

hr {
  border-color: var(--wc-border) !important;
  margin: 1rem 0 !important;
}

code, pre, [data-testid="stCodeBlock"] {
  color: #bae6fd !important;
  background: #07101f !important;
  border-color: var(--wc-border) !important;
}

/* Compact WATERCAST header helpers. */
.wc-topbar {
  position: sticky;
  top: .45rem;
  z-index: 900;
  min-height: 55px;
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 0 0 14px;
  margin-left: 2.8rem;
  padding: 8px 13px;
  border: 1px solid rgba(51, 65, 85, .78);
  border-radius: 9px;
  background: rgba(2, 6, 23, .92);
  box-shadow: 0 8px 30px rgba(0, 0, 0, .24);
  backdrop-filter: blur(16px);
}

.wc-brand {
  display: inline-flex;
  align-items: center;
  flex: 0 0 auto;
  gap: 9px;
  min-width: 180px;
}

.wc-brand__mark {
  width: 34px;
  height: 34px;
  filter: drop-shadow(0 0 8px rgba(34, 211, 238, .22));
}

.wc-brand__name {
  color: var(--wc-text);
  font-size: .93rem;
  font-weight: 800;
  letter-spacing: .19em;
  line-height: 1;
}

.wc-brand__name span { color: var(--wc-cyan); }

.wc-brand__caption {
  margin-top: 5px;
  color: var(--wc-dim);
  font-size: .56rem;
  font-weight: 600;
  letter-spacing: .18em;
  line-height: 1;
}

.wc-topbar__section {
  min-width: 0;
  padding-left: 14px;
  border-left: 1px solid var(--wc-border);
  color: #cbd5e1;
  font-size: .86rem;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.wc-topbar__status {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  flex: 0 0 auto;
  margin-left: auto;
  padding: 5px 9px;
  border: 1px solid var(--wc-border);
  border-radius: 6px;
  color: var(--wc-muted);
  background: rgba(15, 23, 42, .7);
  font-size: .72rem;
  white-space: nowrap;
}

.wc-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--wc-dim);
  box-shadow: 0 0 0 3px rgba(100, 116, 139, .12);
}

.wc-topbar__status--online {
  color: #86efac;
  border-color: rgba(34, 197, 94, .38);
}
.wc-topbar__status--online .wc-status-dot {
  background: #2dd4bf;
  box-shadow: 0 0 9px rgba(45, 212, 191, .72);
}
.wc-topbar__status--offline {
  color: #fca5a5;
  border-color: rgba(239, 68, 68, .42);
}
.wc-topbar__status--offline .wc-status-dot {
  background: var(--wc-red);
  box-shadow: 0 0 9px rgba(239, 68, 68, .65);
}

.wc-panel, .phase-card {
  padding: 16px 18px !important;
  border: 1px solid var(--wc-border) !important;
  border-radius: var(--wc-radius) !important;
  background: var(--wc-surface-soft) !important;
  box-shadow: none !important;
}

.wc-section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--wc-text);
  font-size: .9rem;
  font-weight: 700;
}

.wc-nav-kicker {
  margin: 2px 0 6px;
  color: var(--wc-dim);
  font-size: .61rem;
  font-weight: 800;
  letter-spacing: .16em;
}

.wc-divider {
  height: 1px;
  margin: 11px 0 17px;
  background: linear-gradient(90deg, transparent, var(--wc-border) 7%, var(--wc-border) 93%, transparent);
}

.wc-sample-summary {
  margin: 7px 0 12px;
  padding: 13px 15px;
  border: 1px solid rgba(56, 189, 248, .28);
  border-radius: var(--wc-radius);
  background: linear-gradient(110deg, rgba(8, 47, 73, .42), rgba(15, 23, 42, .82));
}
.wc-sample-summary > span {
  display: inline-flex;
  margin-right: 9px;
  padding: 3px 6px;
  border: 1px solid rgba(34, 211, 238, .34);
  border-radius: 999px;
  color: #67e8f9;
  font-size: .61rem;
  font-weight: 800;
  letter-spacing: .08em;
  vertical-align: middle;
}
.wc-sample-summary > b {
  color: #f8fafc;
  font-size: .92rem;
  vertical-align: middle;
}
.wc-sample-summary > p {
  margin: 7px 0 0;
  color: #94a3b8;
  font-size: .78rem;
}

/* Presentation/EDA helpers used by the API guide and explanatory screens. */
.wc-guide-hero {
  position: relative;
  overflow: hidden;
  margin: 4px 0 16px;
  padding: 22px 24px;
  border: 1px solid rgba(56, 189, 248, .30);
  border-radius: 10px;
  background:
    radial-gradient(circle at 86% 14%, rgba(34, 211, 238, .16), transparent 19rem),
    linear-gradient(118deg, rgba(8, 47, 73, .68), rgba(15, 23, 42, .94) 58%);
  box-shadow: 0 18px 45px rgba(0, 0, 0, .22);
}
.wc-guide-hero::after {
  content: "";
  position: absolute;
  right: -24px;
  bottom: -54px;
  width: 230px;
  height: 110px;
  border: 1px solid rgba(103, 232, 249, .18);
  border-radius: 50%;
  transform: rotate(-7deg);
  pointer-events: none;
}
.wc-guide-hero h2,
.wc-guide-hero h3 { margin: 3px 0 7px !important; color: #f8fafc !important; }
.wc-guide-hero p { max-width: 72rem; margin: 0; color: #cbd5e1; }

.wc-guide-card {
  height: 100%;
  padding: 15px 16px;
  border: 1px solid var(--wc-border);
  border-radius: var(--wc-radius);
  background: linear-gradient(145deg, rgba(15, 23, 42, .9), rgba(8, 15, 31, .88));
  box-shadow: inset 0 1px rgba(255, 255, 255, .015);
}
.wc-guide-card:hover { border-color: rgba(56, 189, 248, .40); }
.wc-guide-card h3,
.wc-guide-card h4 { margin: 0 0 7px !important; color: #e2e8f0 !important; }
.wc-guide-card p,
.wc-guide-card li { color: #94a3b8; font-size: .82rem; line-height: 1.65; }

.wc-term {
  display: inline-flex;
  align-items: center;
  padding: 2px 7px;
  border: 1px solid rgba(34, 211, 238, .28);
  border-radius: 5px;
  color: #67e8f9;
  background: rgba(8, 47, 73, .48);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .76rem;
  font-weight: 700;
}

.wc-explain {
  padding: 12px 14px;
  border: 1px solid var(--wc-border);
  border-left: 3px solid #38bdf8;
  border-radius: 6px;
  color: #cbd5e1;
  background: rgba(15, 23, 42, .72);
  font-size: .82rem;
  line-height: 1.68;
}
.wc-explain strong { color: #e0f2fe; }

.wc-present-note {
  position: relative;
  padding: 12px 14px 12px 42px;
  border: 1px solid rgba(34, 211, 238, .27);
  border-radius: 7px;
  color: #bae6fd;
  background: linear-gradient(100deg, rgba(8, 47, 73, .48), rgba(15, 23, 42, .68));
  font-size: .8rem;
  line-height: 1.6;
}
.wc-present-note::before {
  content: "P";
  position: absolute;
  left: 13px;
  top: 12px;
  width: 20px;
  height: 20px;
  display: grid;
  place-items: center;
  border-radius: 5px;
  color: #083344;
  background: #67e8f9;
  font-size: .68rem;
  font-weight: 900;
}

.wc-step-number {
  width: 28px;
  height: 28px;
  display: inline-grid;
  place-items: center;
  flex: 0 0 28px;
  border: 1px solid rgba(56, 189, 248, .52);
  border-radius: 999px;
  color: #e0f2fe;
  background: rgba(2, 132, 199, .25);
  box-shadow: 0 0 0 4px rgba(14, 165, 233, .06);
  font-size: .75rem;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}

.wc-legend-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 7px 14px;
  color: #94a3b8;
  font-size: .72rem;
}
.wc-legend-row > span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}
.wc-legend-row i {
  width: 7px;
  height: 7px;
  display: inline-block;
  border-radius: 999px;
  background: currentColor;
  box-shadow: 0 0 7px currentColor;
}

.wc-kicker, .eyebrow {
  color: var(--wc-sky) !important;
  font-size: .68rem !important;
  font-weight: 800 !important;
  letter-spacing: .16em !important;
}

.wc-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border: 1px solid var(--wc-border-strong);
  border-radius: 999px;
  color: #cbd5e1;
  background: rgba(30, 41, 59, .72);
  font-size: .72rem;
  font-weight: 600;
}

.wc-risk-normal { color: var(--wc-green) !important; }
.wc-risk-interest { color: var(--wc-yellow) !important; }
.wc-risk-caution { color: var(--wc-orange) !important; }
.wc-risk-alert { color: var(--wc-red) !important; }
.wc-risk-severe { color: var(--wc-purple) !important; }

/* Streamlit controls. */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stCheckbox label,
.stRadio label {
  color: #cbd5e1 !important;
  font-size: .81rem !important;
}

[data-testid="stRadioOption"][data-selected="true"] p {
  color: #7dd3fc !important;
}
[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child {
  border-color: #38bdf8 !important;
  background: #0284c7 !important;
  box-shadow: 0 0 0 3px rgba(56, 189, 248, .12) !important;
}
[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child > div {
  background: #e0f2fe !important;
}

input, textarea,
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="textarea"],
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stDateInput"] > div > div,
[data-testid="stTextInput"] > div > div {
  color: var(--wc-text) !important;
  background: #0b1324 !important;
  border-color: var(--wc-border-strong) !important;
  border-radius: 6px !important;
}

input::placeholder, textarea::placeholder { color: var(--wc-dim) !important; }

input:-webkit-autofill,
input:-webkit-autofill:hover,
input:-webkit-autofill:focus {
  -webkit-text-fill-color: var(--wc-text) !important;
  -webkit-box-shadow: 0 0 0 1000px #0b1324 inset !important;
  caret-color: var(--wc-cyan) !important;
}

input:focus, textarea:focus,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="base-input"]:focus-within {
  border-color: var(--wc-sky-strong) !important;
  box-shadow: 0 0 0 1px rgba(56, 189, 248, .35) !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
  color: var(--wc-text) !important;
  background: var(--wc-surface-raised) !important;
  border-color: var(--wc-border-strong) !important;
}

[role="option"]:hover,
[aria-selected="true"][role="option"] {
  background: rgba(2, 132, 199, .24) !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"],
[data-baseweb="select"] > div,
[data-baseweb="select"] input,
[data-testid="stDateInput"] [data-baseweb="input"],
[data-testid="stDateInput"] [data-baseweb="base-input"],
[data-testid="stTimeInput"] [data-baseweb="input"],
[data-testid="stTimeInput"] [data-baseweb="base-input"] {
  color: var(--wc-text) !important;
  background-color: #0b1324 !important;
  border-color: var(--wc-border-strong) !important;
}

[data-baseweb="select"] svg,
[data-testid="stDateInput"] svg,
[data-testid="stTimeInput"] svg {
  color: var(--wc-muted) !important;
  fill: currentColor !important;
}

[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stVirtualDropdown"],
[data-baseweb="menu"] ul,
[data-baseweb="menu"] li {
  color: var(--wc-text) !important;
  background-color: var(--wc-surface-raised) !important;
  border-color: var(--wc-border-strong) !important;
}

/* BaseWeb renders date pickers in a body-level portal, outside stDateInput. */
[data-st-baseweb-layer-host="true"] [data-baseweb="popover"] {
  color: var(--wc-text) !important;
  background: var(--wc-surface-raised) !important;
  border: 1px solid var(--wc-border-strong) !important;
  border-radius: var(--wc-radius) !important;
  box-shadow: 0 18px 45px rgba(0, 0, 0, .48) !important;
}

[data-baseweb="calendar"],
[data-baseweb="calendar"] > div,
[data-baseweb="calendar"] [role="presentation"] {
  color: var(--wc-text) !important;
  background-color: var(--wc-surface-raised) !important;
}

[data-baseweb="calendar"] button,
[data-baseweb="calendar"] [role="grid"],
[data-baseweb="calendar"] [role="row"] {
  color: #cbd5e1 !important;
  background-color: transparent !important;
}

[data-baseweb="calendar"] [role="gridcell"] {
  color: #cbd5e1 !important;
  background-color: transparent !important;
  border-color: transparent !important;
}
[data-baseweb="calendar"] [role="gridcell"]:hover {
  color: #e0f2fe !important;
  background-color: rgba(2, 132, 199, .22) !important;
}
[data-baseweb="calendar"] [role="gridcell"][aria-label^="Selected"],
[data-baseweb="calendar"] [role="gridcell"][aria-selected="true"] {
  color: #020617 !important;
  background-color: var(--wc-cyan) !important;
  border-radius: 5px !important;
}
[data-baseweb="calendar"] [aria-disabled="true"],
[data-baseweb="calendar"] [role="gridcell"]:not([aria-label]) {
  color: #475569 !important;
}

button:is([data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-tertiary"]),
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
  min-height: 2.25rem;
  border: 1px solid var(--wc-border-strong) !important;
  border-radius: 6px !important;
  color: #dbeafe !important;
  background: #1e293b !important;
  box-shadow: none !important;
  font-weight: 600 !important;
  transition: border-color .15s ease, background .15s ease, color .15s ease,
    transform .15s ease !important;
}

button:is([data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-tertiary"]):hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
  border-color: #0ea5e9 !important;
  color: #e0f2fe !important;
  background: rgba(3, 105, 161, .38) !important;
}

button:is([data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-tertiary"]):active,
[data-testid="stDownloadButton"] button:active { transform: translateY(1px); }

button[data-testid="stBaseButton-primary"],
button[kind="primary"] {
  border-color: #0284c7 !important;
  color: #f0f9ff !important;
  background: linear-gradient(180deg, #0284c7, #0369a1) !important;
}

button:is([data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-tertiary"]):disabled,
[data-testid="stDownloadButton"] button:disabled {
  color: #64748b !important;
  background: rgba(15, 23, 42, .7) !important;
  border-color: var(--wc-border) !important;
  opacity: .72;
}

button:is([data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-tertiary"]) p,
[data-testid="stDownloadButton"] button p,
[data-testid="stFormSubmitButton"] button p {
  color: inherit !important;
}

[data-testid="stFileUploaderDropzone"] {
  color: #cbd5e1 !important;
  background: rgba(8, 47, 73, .18) !important;
  border: 1px dashed #334155 !important;
  border-radius: var(--wc-radius) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--wc-sky-strong) !important;
  background: rgba(8, 47, 73, .3) !important;
}

[data-testid="stSlider"] [role="slider"] {
  border-color: #e0f2fe !important;
  background: var(--wc-sky-strong) !important;
  box-shadow: 0 0 0 4px rgba(14, 165, 233, .16) !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { color: var(--wc-dim) !important; }

[data-testid="stMetric"] {
  min-height: 98px;
  padding: 13px 14px !important;
  border: 1px solid var(--wc-border) !important;
  border-radius: var(--wc-radius) !important;
  color: var(--wc-text) !important;
  background: rgba(15, 23, 42, .8) !important;
  box-shadow: none !important;
}
[data-testid="stMetricLabel"] { color: var(--wc-muted) !important; }
[data-testid="stMetricValue"] {
  color: #f8fafc !important;
  font-variant-numeric: tabular-nums;
  letter-spacing: -.025em;
}

[data-testid="stDataFrame"],
[data-testid="stTable"],
.stTable,
[data-testid="stJson"] {
  overflow: hidden;
  border: 1px solid var(--wc-border) !important;
  border-radius: var(--wc-radius) !important;
  color: var(--wc-text) !important;
  background: #091225 !important;
}

[data-testid="stTable"] th,
[data-testid="stTable"] td,
.stTable th, .stTable td {
  border-color: var(--wc-border) !important;
  color: #cbd5e1 !important;
  background: #0b1324 !important;
}
[data-testid="stTable"] th, .stTable th {
  color: #e2e8f0 !important;
  background: #111c31 !important;
}

/*
 * Streamlit 1.61 st.dataframe/st.data_editor use Glide Data Grid. The visible
 * cells are painted on canvas while editors and menus are regular DOM. The
 * repository-level .streamlit/config.toml supplies Streamlit's native dark
 * theme to the canvas renderer; the rules below cover its DOM overlays and
 * loading surface. A CSS invert filter is deliberately avoided because it can
 * double-transform a native dark grid and make cell values unreadable.
 */
[data-testid="stDataFrame"] {
  position: relative;
  overflow: hidden !important;
  border: 1px solid var(--wc-border-strong) !important;
  border-radius: var(--wc-radius) !important;
  color: var(--wc-text) !important;
  background: #0b1324 !important;
  box-shadow: inset 0 1px rgba(255, 255, 255, .018) !important;
}

[data-testid="stDataFrameResizable"],
[data-testid="stDataFrame"] .dvn-underlay,
[data-testid="stDataFrame"] .dvn-scroller,
[data-testid="stDataFrame"] .dvn-scroll-inner {
  border-color: var(--wc-border-strong) !important;
  background: #0b1324 !important;
}

[data-testid="stDataFrame"] .stDataFrameGlideDataEditor,
[data-testid="stDataFrame"] [class*="gdg-"],
body > [class*="gdg-"] {
  --gdg-accent-color: #22d3ee !important;
  --gdg-accent-fg: #020617 !important;
  --gdg-accent-light: rgba(34, 211, 238, .14) !important;
  --gdg-text-dark: #e2e8f0 !important;
  --gdg-text-medium: #94a3b8 !important;
  --gdg-text-light: #64748b !important;
  --gdg-text-bubble: #cbd5e1 !important;
  --gdg-bg-icon-header: #334155 !important;
  --gdg-fg-icon-header: #e0f2fe !important;
  --gdg-text-header: #cbd5e1 !important;
  --gdg-text-group-header: #94a3b8 !important;
  --gdg-bg-group-header: #111c31 !important;
  --gdg-bg-group-header-hovered: #17243a !important;
  --gdg-text-header-selected: #020617 !important;
  --gdg-bg-cell: #0b1324 !important;
  --gdg-bg-cell-medium: #0f172a !important;
  --gdg-bg-header: #111c31 !important;
  --gdg-bg-header-has-focus: #17243a !important;
  --gdg-bg-header-hovered: #1e293b !important;
  --gdg-bg-bubble: #1e293b !important;
  --gdg-bg-bubble-selected: #164e63 !important;
  --gdg-bg-search-result: rgba(34, 211, 238, .18) !important;
  --gdg-border-color: #1e293b !important;
  --gdg-horizontal-border-color: #1e293b !important;
  --gdg-drilldown-border: rgba(56, 189, 248, .28) !important;
  --gdg-link-color: #38bdf8 !important;
  --gdg-resize-indicator-color: #22d3ee !important;
  --gdg-header-bottom-border-color: #334155 !important;
  --gdg-font-family: Inter, Pretendard, "Noto Sans KR", ui-sans-serif, system-ui, sans-serif !important;
}

[data-testid="stDataFrame"] .dvn-underlay > canvas,
[data-testid="stDataFrame"] canvas[data-testid="data-grid-canvas"] {
  background: #0b1324 !important;
}

/* Accessible DOM grid fallback and browser high-contrast rendering. */
[data-testid="stDataFrame"] table[role="grid"],
[data-testid="stDataFrame"] table[role="grid"] thead,
[data-testid="stDataFrame"] table[role="grid"] tbody,
[data-testid="stDataFrame"] table[role="grid"] tr,
[data-testid="stDataFrame"] table[role="grid"] th,
[data-testid="stDataFrame"] table[role="grid"] td {
  color: #dbeafe !important;
  background: #0b1324 !important;
  border-color: var(--wc-border) !important;
}
[data-testid="stDataFrame"] table[role="grid"] th {
  color: #e0f2fe !important;
  background: #111c31 !important;
}
[data-testid="stDataFrame"] table[role="grid"] [aria-selected="true"] {
  color: #ecfeff !important;
  background: rgba(8, 145, 178, .28) !important;
}

/* Dataframe hover toolbar: visibility, CSV, search, fullscreen and add row. */
[data-testid="stDataFrame"] [data-testid="stElementToolbarButtonContainer"] {
  gap: 2px !important;
  padding: 3px !important;
  border: 1px solid var(--wc-border-strong) !important;
  border-radius: 6px !important;
  background: rgba(7, 16, 31, .96) !important;
  box-shadow: 0 8px 22px rgba(0, 0, 0, .34) !important;
}

[data-testid="stDataFrame"] button[data-testid="stBaseButton-elementToolbar"] {
  width: 29px !important;
  height: 29px !important;
  min-height: 29px !important;
  padding: 6px !important;
  border: 0 !important;
  border-radius: 4px !important;
  color: #94a3b8 !important;
  background: transparent !important;
}
[data-testid="stDataFrame"] button[data-testid="stBaseButton-elementToolbar"]:hover,
[data-testid="stDataFrame"] button[data-testid="stBaseButton-elementToolbar"]:focus-visible {
  color: #67e8f9 !important;
  background: rgba(8, 145, 178, .22) !important;
  box-shadow: 0 0 0 1px rgba(34, 211, 238, .25) !important;
}
[data-testid="stDataFrame"] [data-testid="stElementToolbarButtonIcon"] {
  color: inherit !important;
  fill: currentColor !important;
}

/* Glide search and in-cell editing overlays. */
.gdg-style,
.gdg-search-bar,
[class*="gdg-"] .gdg-search-bar-inner,
[class*="gdg-"] .gdg-clip-region,
[class*="gdg-"] .gdg-growing-entry,
[class*="gdg-"] .glide-select,
[class*="gdg-"] .gdg-multi-select {
  color: var(--gdg-text-dark) !important;
  background: var(--gdg-bg-cell) !important;
  border-color: var(--gdg-border-color) !important;
}

[class*="gdg-"] input,
[class*="gdg-"] textarea,
[class*="gdg-"] select,
[class*="gdg-"] .gdg-input,
[class*="gdg-"] [role="combobox"] {
  color: var(--gdg-text-dark) !important;
  -webkit-text-fill-color: var(--gdg-text-dark) !important;
  caret-color: var(--gdg-accent-color) !important;
  background: var(--gdg-bg-cell) !important;
  border-color: var(--gdg-border-color) !important;
  outline-color: var(--gdg-accent-color) !important;
}
[class*="gdg-"] input::placeholder,
[class*="gdg-"] textarea::placeholder { color: var(--gdg-text-light) !important; }

[class*="gdg-"] .gdg-pill,
[class*="gdg-"] .boe-bubble,
[class*="gdg-"] .doe-bubble {
  color: #cbd5e1 !important;
  background: #1e293b !important;
  border-color: #334155 !important;
}
[class*="gdg-"] .gdg-search-progress { background: var(--gdg-accent-color) !important; }
[class*="gdg-"] .gdg-save-button {
  color: #020617 !important;
  background: var(--gdg-accent-color) !important;
}
[class*="gdg-"] .gdg-close-button {
  color: #cbd5e1 !important;
  background: var(--gdg-bg-header) !important;
}

/* Column visibility/format/statistics and per-cell action menus are portals. */
:is(
  [data-testid="stDataFrameColumnVisibilityMenu"],
  [data-testid="stDataFrameColumnMenu"],
  [data-testid="stDataFrameColumnFormattingMenu"],
  [data-testid="stDataFrameStatisticsMenu"],
  [data-testid="stDataFrameButtonActionMenu"]
) {
  color: var(--wc-text) !important;
  background: #111c31 !important;
  border: 1px solid var(--wc-border-strong) !important;
  border-radius: var(--wc-radius) !important;
  box-shadow: 0 18px 45px rgba(0, 0, 0, .5) !important;
}

:is(
  [data-testid="stDataFrameColumnVisibilityMenu"],
  [data-testid="stDataFrameColumnMenu"],
  [data-testid="stDataFrameColumnFormattingMenu"],
  [data-testid="stDataFrameStatisticsMenu"],
  [data-testid="stDataFrameButtonActionMenu"]
) :is(div, label, span, p, button) {
  color: #cbd5e1 !important;
  border-color: var(--wc-border) !important;
}

:is(
  [data-testid="stDataFrameColumnVisibilityMenu"],
  [data-testid="stDataFrameColumnMenu"],
  [data-testid="stDataFrameColumnFormattingMenu"],
  [data-testid="stDataFrameStatisticsMenu"],
  [data-testid="stDataFrameButtonActionMenu"]
) button:hover {
  color: #67e8f9 !important;
  background: rgba(8, 145, 178, .18) !important;
}

[data-testid="stDataFrameColumnVisibilityMenu"] input[type="checkbox"] + span {
  border-color: #475569 !important;
  background: #0b1324 !important;
}
[data-testid="stDataFrameColumnVisibilityMenu"] input[type="checkbox"]:checked + span,
[data-testid="stDataFrameColumnVisibilityMenu"] span[data-checked="true"],
[data-testid="stDataFrameColumnVisibilityMenu"] span[data-indeterminate="true"] {
  color: #020617 !important;
  border-color: #22d3ee !important;
  background: #22d3ee !important;
}

[data-testid="stDataFrameStatisticsContent"],
[data-testid="stDataFrameStatisticsMetrics"],
[data-testid="stDataFrameStatisticsChart"],
[data-testid="stDataFrameTooltipContent"] {
  color: #cbd5e1 !important;
  background: #0b1324 !important;
  border-color: var(--wc-border) !important;
}

[data-testid="stFullScreenFrame"]:fullscreen,
[data-testid="stFullScreenFrame"]:-moz-full-screen {
  padding: 16px !important;
  color: var(--wc-text) !important;
  background: var(--wc-bg) !important;
}
[data-testid="stFullScreenFrame"]:fullscreen [data-testid="stDataFrame"],
[data-testid="stFullScreenFrame"]:-moz-full-screen [data-testid="stDataFrame"] {
  width: 100% !important;
  height: 100% !important;
  border-color: #334155 !important;
  border-radius: 0 !important;
}

[data-testid="stPlotlyChart"] {
  overflow: hidden;
  padding: 8px 8px 2px;
  border: 1px solid var(--wc-border);
  border-radius: var(--wc-radius);
  background: rgba(8, 15, 31, .86);
}

[data-testid="stExpander"] {
  overflow: hidden;
  border: 1px solid var(--wc-border) !important;
  border-radius: var(--wc-radius) !important;
  color: var(--wc-text) !important;
  background: rgba(15, 23, 42, .68) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--wc-sky) !important; }

[data-baseweb="tab-list"] {
  gap: 5px;
  border-bottom-color: var(--wc-border) !important;
}
[data-baseweb="tab"] {
  height: 2.35rem;
  padding: 0 12px;
  border-radius: 6px 6px 0 0;
  color: var(--wc-muted) !important;
  background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: #7dd3fc !important;
  background: rgba(2, 132, 199, .16) !important;
}

[data-testid="stAlert"] {
  border-radius: var(--wc-radius) !important;
  color: #dbeafe !important;
  background: rgba(15, 23, 42, .92) !important;
  border-color: var(--wc-border-strong) !important;
}

.truth-note {
  padding: 11px 14px !important;
  border: 1px solid rgba(234, 179, 8, .30) !important;
  border-left: 4px solid var(--wc-yellow) !important;
  border-radius: 6px !important;
  color: #fde68a !important;
  background: rgba(113, 63, 18, .18) !important;
}
.ok-note {
  padding: 11px 14px !important;
  border: 1px solid rgba(34, 197, 94, .28) !important;
  border-left: 4px solid var(--wc-green) !important;
  border-radius: 6px !important;
  color: #bbf7d0 !important;
  background: rgba(20, 83, 45, .2) !important;
}
.method { color: #67e8f9 !important; }

[data-testid="stSidebar"] {
  color: var(--wc-text) !important;
  background: #07101f !important;
  border-right: 1px solid var(--wc-border) !important;
}
[data-testid="stSidebarHeader"] {
  height: 3.1rem !important;
  min-height: 3.1rem !important;
}
[data-testid="stSidebarCollapseButton"] {
  visibility: visible !important;
}
[data-testid="stSidebarContent"] { padding-top: .8rem !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label { color: #cbd5e1 !important; }

[data-testid="stSpinner"] { color: var(--wc-sky) !important; }
[data-testid="stToast"] {
  color: var(--wc-text) !important;
  background: #111c31 !important;
  border: 1px solid var(--wc-border-strong) !important;
}

/* Short, non-blocking route transition with an original droplet/forecast mark. */
.wc-transition {
  position: fixed;
  inset: 0;
  z-index: 999999;
  display: grid;
  place-items: center;
  pointer-events: none;
  isolation: isolate;
  background:
    radial-gradient(circle at 50% 52%, rgba(8, 47, 73, .58), transparent 23rem),
    rgba(2, 6, 23, .94);
  animation: wc-transition-screen .62s cubic-bezier(.22, .75, .25, 1) both;
}

.wc-transition__content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 11px;
  animation: wc-transition-rise .62s cubic-bezier(.2, .8, .25, 1) both;
}

.wc-transition__halo {
  position: relative;
  width: 86px;
  height: 86px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(56, 189, 248, .42);
  border-radius: 999px;
  background: rgba(8, 47, 73, .42);
  box-shadow:
    0 0 0 9px rgba(14, 165, 233, .035),
    0 0 34px rgba(34, 211, 238, .24),
    inset 0 0 24px rgba(56, 189, 248, .08);
}

.wc-transition__halo::after {
  content: "";
  position: absolute;
  left: 50%;
  top: 50%;
  width: 112px;
  height: 32px;
  border: 1px solid rgba(56, 189, 248, .25);
  border-radius: 50%;
  transform: translate(-50%, -50%) scale(.5);
  animation: wc-transition-ripple .62s ease-out both;
}

.wc-transition__mark {
  width: 58px;
  height: 58px;
  overflow: visible;
  filter: drop-shadow(0 0 9px rgba(34, 211, 238, .34));
}

.wc-transition__forecast-line {
  stroke-dasharray: 94;
  stroke-dashoffset: 94;
  animation: wc-transition-draw .52s ease-out .04s forwards;
}

.wc-transition__brand {
  color: #e2e8f0;
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .30em;
  margin-right: -.30em;
}
.wc-transition__brand span { color: var(--wc-cyan); }
.wc-transition__phase {
  color: var(--wc-muted);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .08em;
}

@keyframes wc-transition-screen {
  0% { opacity: 0; visibility: visible; }
  13% { opacity: 1; }
  70% { opacity: 1; }
  99% { visibility: visible; }
  100% { opacity: 0; visibility: hidden; }
}
@keyframes wc-transition-rise {
  0% { opacity: 0; transform: translateY(8px) scale(.92); }
  22% { opacity: 1; transform: translateY(0) scale(1); }
  74% { opacity: 1; transform: translateY(0) scale(1); }
  100% { opacity: 0; transform: translateY(-3px) scale(1.015); }
}
@keyframes wc-transition-ripple {
  0% { opacity: 0; transform: translate(-50%, -50%) scale(.35); }
  28% { opacity: .8; }
  100% { opacity: 0; transform: translate(-50%, -50%) scale(1.55); }
}
@keyframes wc-transition-draw { to { stroke-dashoffset: 0; } }

@media (max-width: 760px) {
  .block-container, [data-testid="stMainBlockContainer"] {
    padding-left: .65rem !important;
    padding-right: .65rem !important;
  }
  .wc-topbar { gap: 9px; padding-inline: 10px; }
  .wc-brand { min-width: 0; }
  .wc-brand__caption, .wc-topbar__section { display: none; }
  .wc-brand__name { font-size: .82rem; }
  .wc-topbar__status { max-width: 42%; overflow: hidden; text-overflow: ellipsis; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
  }
  .wc-transition {
    animation: wc-transition-reduced .14s ease-out both !important;
  }
  .wc-transition__content,
  .wc-transition__halo::after,
  .wc-transition__forecast-line {
    animation: none !important;
  }
  @keyframes wc-transition-reduced {
    0% { opacity: .55; visibility: visible; }
    99% { visibility: visible; }
    100% { opacity: 0; visibility: hidden; }
  }
}
</style>
"""


def _mark_svg(*, class_name: str = "wc-brand__mark") -> str:
    """Return the original WATERCAST droplet/forecast mark."""

    return f"""
<svg class="{escape(class_name, quote=True)}" viewBox="0 0 64 64"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
  <defs>
    <linearGradient id="wc-mark-gradient" x1="12" y1="8" x2="52" y2="56"
                    gradientUnits="userSpaceOnUse">
      <stop stop-color="#67e8f9"/>
      <stop offset="1" stop-color="#0284c7"/>
    </linearGradient>
  </defs>
  <path d="M32 5C24 16 13 26 13 39c0 11 8.5 20 19 20s19-9 19-20C51 26 40 16 32 5Z"
        fill="rgba(8,47,73,.72)" stroke="url(#wc-mark-gradient)" stroke-width="2.6"/>
  <path class="wc-transition__forecast-line"
        d="M19 42c4-5 7 4 11-1s7-8 11-3l5-6"
        fill="none" stroke="#67e8f9" stroke-width="2.7" stroke-linecap="round"
        stroke-linejoin="round"/>
  <circle cx="46" cy="32" r="2.4" fill="#e0f2fe"/>
</svg>""".strip()


def inject_theme() -> None:
    """Inject the WATERCAST CSS into the current Streamlit page."""

    import streamlit as st

    st.markdown(THEME_CSS, unsafe_allow_html=True)


def compact_header_html(
    section_title: str,
    status_label: str = "API 상태 확인 전",
    connected: bool | None = None,
) -> str:
    """Return the compact sticky WATERCAST operator header.

    ``connected`` controls the status-dot style: ``True`` is online, ``False``
    is offline, and ``None`` is the neutral/unchecked state.
    """

    status_modifier = (
        "wc-topbar__status--online"
        if connected is True
        else "wc-topbar__status--offline"
        if connected is False
        else "wc-topbar__status--unknown"
    )
    return f"""
<div class="wc-topbar">
  <div class="wc-brand">
    {_mark_svg()}
    <div>
      <div class="wc-brand__name">WATER<span>CAST</span></div>
      <div class="wc-brand__caption">SATELLITE ANALYTICS CONSOLE</div>
    </div>
  </div>
  <div class="wc-topbar__section">{escape(str(section_title))}</div>
  <div class="wc-topbar__status {status_modifier}">
    <span class="wc-status-dot" aria-hidden="true"></span>
    <span>{escape(str(status_label))}</span>
  </div>
</div>""".strip()


def transition_overlay_html(phase_title: str) -> str:
    """Return a 0.62-second, pointer-safe phase transition overlay."""

    safe_title = escape(str(phase_title))
    accessible_label = escape(f"{phase_title} 화면으로 이동 중", quote=True)
    return f"""
<div class="wc-transition" role="status" aria-live="polite"
     aria-label="{accessible_label}">
  <div class="wc-transition__content">
    <div class="wc-transition__halo">
      {_mark_svg(class_name="wc-transition__mark")}
    </div>
    <div class="wc-transition__brand">WATER<span>CAST</span></div>
    <div class="wc-transition__phase">{safe_title}</div>
  </div>
</div>""".strip()


__all__ = [
    "THEME_CSS",
    "compact_header_html",
    "inject_theme",
    "transition_overlay_html",
]
