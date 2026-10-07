"""Onglet 1 — Univers d'investissement, statistiques descriptives et diversification."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from src import ui
from src.analytics import diversification_summary, hierarchical_clusters, rolling_correlation, summary_table
from src.backtest import run_backtest, static_target
from src.config import BENCHMARK_NAME, BENCHMARK_WEIGHTS, EQUAL_WEIGHT_NAME, UNIVERSE, BacktestConfig
from src.data import MarketData
from src.metrics import performance


@st.cache_data(show_spinner=False)
def _benchmarks_full_sample(fp: str, _md: MarketData) -> pd.DataFrame:
    cfg = BacktestConfig(buffer=0.0, execution_lag=0)
    rows = {}
    n = len(_md.prices.columns)
    for name, w in ((BENCHMARK_NAME, BENCHMARK_WEIGHTS), (EQUAL_WEIGHT_NAME, {t: 1 / n for t in _md.prices.columns})):
        res = run_backtest(_md.returns, _md.rf, static_target(w), cfg, start=_md.returns.index[0])
        m = performance(res.returns, _md.rf)
        rows[name] = {k: m[k] for k in ("Rendement annualisé", "Volatilité annualisée", "Ratio de Sharpe",
                                         "Max Drawdown", "Ratio de Calmar")}
    return pd.DataFrame(rows).T


def _dendrogram_figure(clusters: dict, labels: list[str]) -> go.Figure:
    d = dendrogram(clusters["linkage"], labels=labels, no_plot=True)
    fig = go.Figure()
    for xs, ys in zip(d["icoord"], d["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=ui.INK_2, width=1.5),
                                 hoverinfo="skip", showlegend=False))
    Z = clusters["linkage"]
    k = clusters["k"]
    cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2 if k > 1 else Z[-1, 2]
    fig.add_hline(y=cut, line=dict(color=ui.BENCHMARK, width=1, dash="dash"),
                  annotation_text=f"coupe : {k} groupes", annotation_position="top right",
                  annotation_font=dict(size=11, color=ui.INK_2))
    ticks = [5 + 10 * i for i in range(len(d["ivl"]))]
    fig.update_layout(title="Dendrogramme (liaison de Ward, distance √(½(1 − ρ)))", hovermode=False,
                      xaxis=dict(tickvals=ticks, ticktext=d["ivl"], showgrid=False),
                      yaxis=dict(title="Distance de fusion"))
    return fig


def _corr_heatmap(corr: pd.DataFrame, order: list[str]) -> go.Figure:
    c = corr.loc[order, order]
    fig = go.Figure(go.Heatmap(
        z=c.values, x=order, y=order, zmin=-1, zmax=1, colorscale=ui.DIVERGING,
        text=np.round(c.values, 2), texttemplate="%{text:.2f}", textfont=dict(size=11),
        hovertemplate="%{y} / %{x} : %{z:.2f}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=10)),
    ))
    fig.update_layout(title="Matrice de corrélation (ordonnée par clustering)", hovermode="closest",
                      yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False))
    return fig


def render(md: MarketData, fp: str) -> None:
    prices, lr = md.prices, md.log_returns
    colors = ui.asset_colors(list(prices.columns))

    st.markdown("## Univers d'investissement")
    st.markdown(
        "Huit ETF liquides, un par prime de risque structurellement distincte : actions "
        "(US, développées hors US, émergentes), duration (intermédiaire, longue), or, matières "
        "premières et immobilier coté. Les prix sont ajustés des dividendes : toutes les "
        "statistiques sont en rendement total, nettes des frais de gestion des ETF.")
    stats = summary_table(prices, md.rf)
    meta = pd.DataFrame(UNIVERSE).T.rename(columns={"name": "Exposition", "asset_class": "Classe", "role": "Rôle"})
    table = meta[["Exposition", "Classe"]].join(stats)
    table.index.name = "Ticker"
    ui.show_table(table)
    st.caption(f"Période : {prices.index[0]:%d/%m/%Y} – {prices.index[-1]:%d/%m/%Y} · {len(prices)} semaines · "
               "Sharpe calculé en excès du taux des T-Bills 3 mois · Jarque-Bera p : test de normalité des "
               "log-rendements hebdomadaires.")

    st.markdown("### Benchmarks de référence")
    st.markdown(
        f"**Benchmark principal : {BENCHMARK_NAME}**, rééquilibré toutes les 4 semaines, mêmes coûts de "
        "transaction que la stratégie. C'est le portefeuille de politique d'investissement standard d'un "
        "investisseur diversifié : il partage le même univers, la même contrainte long-only sans levier et "
        "un niveau de risque comparable. Il est investissable, transparent et sans paramètre estimé. "
        f"**Référence secondaire : {EQUAL_WEIGHT_NAME}**. La pondération naïve est un adversaire notoirement "
        "difficile pour les optimiseurs (DeMiguel, Garlappi & Uppal, 2009). La battre isole la valeur "
        "ajoutée de la modélisation, à univers identique.")
    ui.show_table(_benchmarks_full_sample(fp, md))

    st.markdown("## Évolution des prix")
    sel = st.multiselect("Actifs affichés", list(prices.columns), default=list(prices.columns), key="u_assets")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for t in sel:
            fig.add_trace(go.Scatter(x=prices.index, y=100 * prices[t] / prices[t].iloc[0], name=t,
                                     line=dict(color=colors[t], width=1.6),
                                     hovertemplate=f"{t} : %{{y:.0f}}<extra></extra>"))
        fig.update_layout(title="Performance cumulée (base 100, échelle log)", yaxis_type="log")
        ui.chart(fig, 400)
    with c2:
        vol = lr.rolling(26).std() * np.sqrt(52)
        fig = go.Figure()
        for t in sel:
            fig.add_trace(go.Scatter(x=vol.index, y=vol[t], name=t, line=dict(color=colors[t], width=1.4),
                                     hovertemplate=f"{t} : %{{y:.1%}}<extra></extra>"))
        fig.update_layout(title="Volatilité réalisée glissante 26 semaines (annualisée)", yaxis_tickformat=".0%")
        ui.chart(fig, 400)

    st.markdown("## Diversification structurelle")
    period = st.radio("Fenêtre d'estimation", ["Historique complet", "5 dernières années", "2 dernières années"],
                      horizontal=True, key="u_period")
    weeks = {"Historique complet": len(lr), "5 dernières années": 260, "2 dernières années": 104}[period]
    sample = lr.tail(weeks)
    corr = sample.corr()
    cl = hierarchical_clusters(corr)
    div = diversification_summary(sample)

    ui.kpi_row([
        ("Corrélation moyenne", ui.num(div["Corrélation moyenne"]), "hors diagonale"),
        ("Nombre effectif de paris", ui.num(div["Nombre effectif de paris"], 1), f"sur {len(corr)} actifs"),
        ("Ratio de diversification", ui.num(div["Ratio de diversification (1/N)"]), "portefeuille 1/N"),
        ("Part de la 1re composante", ui.pct(div["Variance expliquée par la 1re composante"]), "ACP de la corrélation"),
        ("Groupes identifiés", str(cl["k"]), f"silhouette {cl['silhouette'].max():.2f}"),
    ])
    st.write("")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        ui.chart(_corr_heatmap(corr, cl["order"]), 460)
    with c2:
        ui.chart(_dendrogram_figure(cl, list(corr.index)), 460)

    groups = cl["labels"].groupby(cl["labels"]).apply(lambda s: ", ".join(s.index))
    intra = {}
    for g, members in cl["labels"].groupby(cl["labels"]):
        m = list(members.index)
        sub = corr.loc[m, m].values
        intra[g] = sub[~np.eye(len(m), dtype=bool)].mean() if len(m) > 1 else np.nan
    cluster_tbl = pd.DataFrame({"Actifs": groups, "Corrélation intra-groupe": pd.Series(intra)})
    cluster_tbl.index = [f"Groupe {g}" for g in cluster_tbl.index]
    ui.show_table(cluster_tbl)
    ui.note(
        "Lecture : le clustering hiérarchique regroupe les actifs dont les rendements co-varient. Un univers "
        "bien diversifié présente plusieurs groupes faiblement corrélés entre eux. Le nombre effectif de paris "
        "(exponentielle de l'entropie du spectre de corrélation) mesure combien de sources de risque réellement "
        "indépendantes l'univers contient. Il vaut N si les actifs sont décorrélés, 1 s'ils sont parfaitement "
        "corrélés. Le nombre de groupes est choisi par maximisation du score de silhouette.")

    st.markdown("### Stabilité de la diversification")
    fig = go.Figure()
    for a, b, col in (("SPY", "IEF", ui.STRATEGY), ("SPY", "GLD", ui.EQUAL_WEIGHT), ("SPY", "EEM", ui.BENCHMARK)):
        rc = rolling_correlation(lr, a, b, 52)
        fig.add_trace(go.Scatter(x=rc.index, y=rc, name=f"{a} / {b}", line=dict(color=col, width=1.8),
                                 hovertemplate=f"{a}/{b} : %{{y:.2f}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=ui.INK_3, width=1))
    fig.update_layout(title="Corrélations glissantes 52 semaines", yaxis=dict(range=[-1, 1]))
    ui.chart(fig, 360)
    ui.note(
        "Limite : les corrélations ne sont pas stables. La corrélation actions / obligations, négative "
        "pendant la majeure partie de 2000-2021, est redevenue positive lors du choc inflationniste de 2022. "
        "Dans ce régime, la duration ne protège plus un portefeuille actions. C'est précisément ce que "
        "le filtre de tendance et la poche de liquidité sont conçus à absorber.")
