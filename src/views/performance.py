"""Onglet 3 — Backtest et performance."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import BENCHMARK_NAME, EQUAL_WEIGHT_NAME, Config
from src.data import MarketData
from src.metrics import calendar_returns, drawdown, performance, relative, rolling_sharpe

KEYS = ["Rendement annualisé", "Volatilité annualisée", "Ratio de Sharpe", "Max Drawdown", "Ratio de Calmar"]


def render(md: MarketData, run, cfg: Config) -> None:
    s, b, e = run.strategy, run.benchmark, run.equal_weight
    series = {"Stratégie": s, BENCHMARK_NAME: b, EQUAL_WEIGHT_NAME: e}
    colors = {"Stratégie": ui.STRATEGY, BENCHMARK_NAME: ui.BENCHMARK, EQUAL_WEIGHT_NAME: ui.EQUAL_WEIGHT}
    perf = {k: performance(v.returns, md.rf) for k, v in series.items()}
    ms, mb = perf["Stratégie"], perf[BENCHMARK_NAME]

    st.markdown(f"## Performance hors échantillon · {s.equity.index[0]:%d/%m/%Y} – {s.equity.index[-1]:%d/%m/%Y}")
    st.markdown(f'<div class="subtitle">Walk-forward intégral, nets de coûts ({cfg.backtest.cost_bps:g} bps), '
                f'{cfg.backtest.n_tranches} tranches, exécution à t + {cfg.backtest.execution_lag} semaine. '
                f'Écarts affichés vs {BENCHMARK_NAME}.</div>', unsafe_allow_html=True)

    def d_pct(k):
        return f"{(ms[k] - mb[k]) * 100:+.1f} pts vs benchmark"

    def d_num(k):
        return f"{ms[k] - mb[k]:+.2f} vs benchmark"

    ui.kpi_row([
        ("Rendement annualisé", ui.pct(ms[KEYS[0]]), d_pct(KEYS[0])),
        ("Volatilité annualisée", ui.pct(ms[KEYS[1]]), d_pct(KEYS[1])),
        ("Ratio de Sharpe", ui.num(ms[KEYS[2]]), d_num(KEYS[2])),
        ("Max Drawdown", ui.pct(ms[KEYS[3]]), d_pct(KEYS[3])),
        ("Ratio de Calmar", ui.num(ms[KEYS[4]]), d_num(KEYS[4])),
    ])
    st.write("")

    fig = go.Figure()
    for name, res in series.items():
        fig.add_trace(go.Scatter(x=res.equity.index, y=100 * res.equity, name=name,
                                 line=dict(color=colors[name], width=2.4 if name == "Stratégie" else 1.6),
                                 hovertemplate=f"{name} : %{{y:.1f}}<extra></extra>"))
    fig.update_layout(title="Valeur du portefeuille (base 100, échelle log)", yaxis_type="log")
    ui.chart(fig, 430)

    fig = go.Figure()
    for name in ("Stratégie", BENCHMARK_NAME):
        dd = drawdown(series[name].equity)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, line=dict(color=colors[name], width=1.6),
                                 fill="tozeroy" if name == "Stratégie" else None,
                                 fillcolor="rgba(42,120,214,0.10)",
                                 hovertemplate=f"{name} : %{{y:.1%}}<extra></extra>"))
    fig.update_layout(title="Drawdown", yaxis_tickformat=".0%")
    ui.chart(fig, 260)

    st.markdown("### Tableau de performance")
    tbl = pd.DataFrame(perf).T[KEYS + ["Ratio de Sortino", "t-stat Sharpe (Lo 2002)"]]
    rel = pd.DataFrame({k: relative(v.returns, b.returns) for k, v in series.items() if k != BENCHMARK_NAME}).T
    tbl = tbl.join(rel).astype(float)
    years = len(s.returns) / 52
    tbl.loc["Stratégie", "Turnover annuel"] = s.turnover.sum() / years
    tbl.loc["Stratégie", "Exposition moyenne"] = s.weights.sum(axis=1).mean()
    ui.show_table(tbl, pct_cols=["Tracking error"])
    st.caption("Sharpe : rendement excédentaire moyen sur T-Bills × 52 / volatilité. Calmar : rendement annualisé / "
               "|drawdown maximal|. t-stat Sharpe : erreur-type de Lo (2002) sous hypothèse i.i.d. Bêta, tracking "
               f"error et ratio d'information calculés contre le {BENCHMARK_NAME}.")

    c1, c2 = st.columns([1.5, 1])
    with c1:
        w = s.weights.copy()
        w["Cash (T-Bills)"] = s.cash
        palette = ui.asset_colors(list(s.weights.columns))
        palette["Cash (T-Bills)"] = ui.CASH
        fig = go.Figure()
        for col in w.columns:
            fig.add_trace(go.Scatter(x=w.index, y=w[col], name=col, stackgroup="one", mode="none",
                                     fillcolor=palette[col], hovertemplate=f"{col} : %{{y:.0%}}<extra></extra>"))
        fig.update_layout(title="Allocation agrégée (toutes tranches)", yaxis=dict(tickformat=".0%", range=[0, 1]))
        ui.chart(fig, 380)
    with c2:
        cal = pd.DataFrame({k: calendar_returns(v.equity) for k, v in series.items()})
        cal.index.name = "Année"
        st.markdown("**Rendements par année civile**")
        st.dataframe(cal.style.format(lambda v: ui.pct(v, 1)), width="stretch", height=380)

    fig = go.Figure()
    for name in ("Stratégie", BENCHMARK_NAME):
        rs = rolling_sharpe(series[name].returns, md.rf, 104)
        fig.add_trace(go.Scatter(x=rs.index, y=rs, name=name, line=dict(color=colors[name], width=1.6),
                                 hovertemplate=f"{name} : %{{y:.2f}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=ui.INK_3, width=1))
    fig.update_layout(title="Ratio de Sharpe glissant sur 2 ans")
    ui.chart(fig, 280)

    export = pd.DataFrame({f"{k} — valeur": v.equity for k, v in series.items()}).join(
        s.weights.add_prefix("Poids "))
    st.download_button("Exporter les séries (CSV)", export.to_csv().encode("utf-8"),
                       file_name="backtest_safety_first.csv", mime="text/csv")
