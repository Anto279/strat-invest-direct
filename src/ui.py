"""
Couche de présentation : thème graphique, styles et helpers de mise en forme.

Charte : fond clair neutre, encre quasi noire, une seule couleur d'accent
(bleu) pour la stratégie, orange pour le benchmark principal, vert d'eau pour
le 1/N. Palette catégorielle à ordre fixe validée pour les daltonismes
(une couleur suit toujours le même actif). Corrélations : palette divergente
bleu / gris / rouge centrée sur zéro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK = "#1a1a19"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e9e8e3"
SURFACE = "#fcfcfb"
CASH = "#c9c7bf"

STRATEGY = "#2a78d6"
BENCHMARK = "#eb6834"
EQUAL_WEIGHT = "#1baf7a"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIVERGING = [[0.0, "#b5302f"], [0.25, "#e8908f"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#184f95"]]
SEQUENTIAL = [[0.0, "#f0efec"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#256abf"], [1.0, "#0d366b"]]

FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"


def asset_colors(tickers: list[str]) -> dict[str, str]:
    return {t: CATEGORICAL[i % len(CATEGORICAL)] for i, t in enumerate(tickers)}


def register_template() -> None:
    axis = dict(gridcolor=GRID, linecolor="#cfcdc6", zerolinecolor="#cfcdc6", ticks="outside",
                tickcolor="#cfcdc6", ticklen=4, title=dict(font=dict(size=12, color=INK_2)),
                tickfont=dict(size=11, color=INK_2))
    pio.templates["institutional"] = go.layout.Template(layout=go.Layout(
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        colorway=CATEGORICAL,
        title=dict(font=dict(size=14, color=INK), x=0, xanchor="left", y=0.98),
        xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color=INK_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor="#cfcdc6", font=dict(family=FONT, size=12, color=INK)),
        hovermode="x unified",
        margin=dict(l=8, r=8, t=68, b=8),
    ))
    pio.templates.default = "institutional"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stMarkdownContainer"] h1, h1 { font-weight: 600 !important; font-size: 1.6rem !important;
     letter-spacing: -0.01em; color: #1a1a19; padding-bottom: 0 !important; }
[data-testid="stMarkdownContainer"] h2 { font-weight: 600 !important; font-size: 1.12rem !important; color: #1a1a19;
     margin-top: 1.8rem; padding: 0 0 .4rem 0 !important; border-bottom: 1px solid #e3e2dc; }
[data-testid="stMarkdownContainer"] h3 { font-weight: 600 !important; font-size: .95rem !important; color: #1a1a19;
     margin-top: 1.1rem; padding: .2rem 0 !important; }
p, li { color: #3a3936; font-size: 0.92rem; line-height: 1.55; }
.subtitle { color: #6b6a65; font-size: .86rem; margin-top: .15rem; margin-bottom: 1.2rem; }
.eyebrow { text-transform: uppercase; letter-spacing: .08em; font-size: .70rem; color: #8a8983; font-weight: 600; }
.kpi { border-top: 2px solid #1a1a19; padding-top: .45rem; }
.kpi .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: #6b6a65; font-weight: 600; }
.kpi .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.45rem; color: #1a1a19; font-weight: 500; }
.kpi .delta { font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: #6b6a65; }
.note { border-left: 2px solid #c9c7bf; padding: .35rem .8rem; color: #52514e; font-size: .86rem; background: #f6f5f1; }
.finding { border-left: 2px solid #2a78d6; padding: .35rem .8rem; color: #1a1a19; font-size: .88rem; background: #f3f7fc; }
[data-testid="stDataFrame"] { font-family: 'IBM Plex Mono', monospace; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #e3e2dc; }
.stTabs [data-baseweb="tab"] { font-weight: 500; padding-left: 0; padding-right: 0; }
section[data-testid="stSidebar"] { background: #f6f5f1; border-right: 1px solid #e3e2dc; }
section[data-testid="stSidebar"] h2 { border: none; font-size: .78rem; text-transform: uppercase;
     letter-spacing: .08em; color: #6b6a65; margin-top: 1.2rem; }
#MainMenu, footer { visibility: hidden; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str | None = None) -> str:
    d = f'<div class="delta">{delta}</div>' if delta else ""
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>{d}</div>'


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for c, (label, value, delta) in zip(cols, items):
        c.markdown(kpi(label, value, delta), unsafe_allow_html=True)


def note(text: str, kind: str = "note") -> None:
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def pct(x, d: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f} %"


def num(x, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def pvalue(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


PCT_COLS = {"Rendement annualisé", "Volatilité annualisée", "Max Drawdown", "Semaines positives",
            "Tracking error", "Turnover annuel", "Coûts annuels", "Exposition moyenne", "Buffer",
            "Taux de bon sens", "R² OOS vs moyenne hist.", "R² OOS vs zéro", "Poids", "Borne haute",
            "Prior", "Prévision", "μ retenu", "Volatilité (horizon)", "Score de tendance"}


def _formatters(df: pd.DataFrame, pct_cols=None, dec: int = 2) -> dict:
    """Formateur par colonne numérique : %, p-value, entier ou décimal."""
    pct_cols = set(pct_cols or []) | PCT_COLS
    fmt = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        name = str(c)
        if name in pct_cols:
            fmt[c] = lambda v: pct(v, 1)
        elif "p-value" in name or name.endswith(" p"):
            fmt[c] = pvalue
        elif pd.api.types.is_integer_dtype(df[c]):
            fmt[c] = lambda v: f"{int(v):d}"
        else:
            fmt[c] = lambda v, d=dec: num(v, d)
    return fmt


def show_table(df: pd.DataFrame, pct_cols=None, dec: int = 2, height: int | None = None, index: bool = True):
    """Tableau formaté. Les valeurs sont converties en texte : Streamlit affiche « None »
    pour les NaN même lorsqu'un Styler les formate ; on garantit ainsi « — »."""
    kwargs = {"width": "stretch", "hide_index": not index}
    if height:
        kwargs["height"] = height
    fmt = _formatters(df, pct_cols, dec)
    shown = df.astype(object).copy()
    for c, f in fmt.items():
        shown[c] = [f(v) if pd.notna(v) else "—" for v in df[c]]
    st.dataframe(shown.style.set_properties(subset=list(fmt), **{"text-align": "right"}), **kwargs)


def chart(fig: go.Figure, height: int = 380, legend_bottom: bool | None = None) -> None:
    # Au-delà de 4 séries, la légende passe sous le graphique pour ne jamais chevaucher le titre.
    if legend_bottom is None:
        legend_bottom = sum(1 for t in fig.data if t.showlegend is not False) > 4
    if legend_bottom:
        has_xtitle = bool(fig.layout.xaxis.title.text)
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2 if has_xtitle else -0.1,
                                      xanchor="left", x=0), margin=dict(b=80 if has_xtitle else 60))
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        "toImageButtonOptions": {"format": "svg", "scale": 2},
    })
