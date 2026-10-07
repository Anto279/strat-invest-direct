"""Onglet 2 — Recherche et preuves statistiques : pipeline ARIMA → XGBoost → GARCH → Markowitz → exécution."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf

from src import ui
from src.config import Config
from src.data import MarketData
from src.features import FEATURE_DOC, FEATURES
from src.models import garch as G
from src.models.arima import (coefficient_table, fit_arma, integration_order, order_grid,
                              residual_diagnostics)
from src.pipeline import ablation, forecast_evaluation, sensitivity, tranche_dispersion
from src.stats_tests import diebold_mariano
from src.trend import conditional_stats, sma


# ---------------------------------------------------------------------------
# Calculs mis en cache (clé : empreinte des données + paramètres)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Estimation de la grille ARIMA…")
def _arima_study(fp: str, ticker: str, split: pd.Timestamp, _md: MarketData, max_p: int, max_q: int):
    lp = np.log(_md.prices[ticker])
    y = 100 * lp.diff().dropna()
    y_tr = y.loc[:split]
    d, stat_tbl = integration_order(lp.loc[:split])
    grid = order_grid(y_tr.values, max_p, max_q)
    best_bic = grid.loc[grid.BIC.idxmin(), ["p", "q"]].astype(int).tolist()
    best_aicc = grid.loc[grid.AICc.idxmin(), ["p", "q"]].astype(int).tolist()
    res = fit_arma(y_tr.values, tuple(best_bic))
    full = res.apply(y.values)                       # paramètres gelés, filtrage sur tout l'échantillon
    burn = max(best_bic) + 1
    resid_full = pd.Series(full.resid, index=y.index).iloc[burn:]
    resid_tr = resid_full.loc[:split]
    return {
        "d": d, "stationarity": stat_tbl, "grid": grid, "bic": tuple(best_bic), "aicc": tuple(best_aicc),
        "coef": coefficient_table(res), "diag": residual_diagnostics(resid_tr.values),
        "y_train": y_tr, "resid_train": resid_tr, "resid_full": resid_full,
        "acf": acf(y_tr, nlags=26), "pacf": pacf(y_tr, nlags=26), "resid_acf": acf(resid_tr, nlags=26),
    }


@st.cache_data(show_spinner="Estimation des modèles GARCH…")
def _garch_study(fp: str, ticker: str, split: pd.Timestamp, _resid_full: pd.Series):
    e = _resid_full
    split_idx = int(e.index.searchsorted(split, side="right"))
    e_tr, e_te = e.iloc[:split_idx], e.iloc[split_idx:]
    train_specs = G.compare_specs(e_tr)
    res = G.fit_garch(e_tr.values)
    oos = G.out_of_sample(e, split_idx)
    test_specs = G.compare_specs(e_te)
    params = None
    if res is not None:
        params = pd.DataFrame({"Estimation": res.params, "Erreur std.": res.std_err,
                               "t": res.tvalues, "p-value": res.pvalues})
    return {"train_specs": train_specs, "params": params,
            "std_diag": G.standardized_diagnostics(res) if res is not None else None,
            "oos": {k: v for k, v in oos.items() if k not in ("train_result", "test_fit")},
            "test_specs": test_specs, "n_train": len(e_tr), "n_test": len(e_te)}


@st.cache_data(show_spinner="Évaluation hors échantillon des prévisions…")
def _forecast_eval(fp: str, key: str, _signals: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return forecast_evaluation(_signals, horizon)


@st.cache_data(show_spinner="Backtests d'ablation (une composante retirée à la fois)…")
def _ablation(fp: str, key: str, _md, _signals, _cfg: Config) -> pd.DataFrame:
    return ablation(_md, _signals, _cfg)


@st.cache_data(show_spinner="Backtests de sensibilité (buffer × coûts)…")
def _sensitivity(fp: str, key: str, _md, _signals, _cfg: Config) -> pd.DataFrame:
    return sensitivity(_md, _signals, _cfg)


@st.cache_data(show_spinner=False)
def _trend_stats(fp: str, windows: tuple, horizon: int, _md: MarketData) -> pd.DataFrame:
    from src.trend import trend_score
    score = trend_score(_md.prices, windows)
    rows = {}
    for t in _md.prices.columns:
        cs = conditional_stats(_md.prices[t], score[t], horizon)
        up, down = cs.iloc[0], cs.iloc[1]
        rows[t] = {"Temps en tendance haussière": (score[t] == 1).mean(),
                   "Rdt ann. | haussier": up["Rendement annualisé"], "Rdt ann. | baissier": down["Rendement annualisé"],
                   "Vol ann. | haussier": up["Volatilité annualisée"], "Vol ann. | baissier": down["Volatilité annualisée"],
                   "Welch t": cs.attrs.get("welch_t", np.nan), "Welch p-value": cs.attrs.get("welch_p", np.nan),
                   "Levene p-value": cs.attrs.get("levene_p", np.nan)}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Graphiques utilitaires
# ---------------------------------------------------------------------------
def _corr_bars(values: np.ndarray, n: int, title: str) -> go.Figure:
    lags = np.arange(1, len(values))
    band = 1.96 / np.sqrt(n)
    fig = go.Figure(go.Bar(x=lags, y=values[1:], marker_color=ui.STRATEGY, width=0.5,
                           hovertemplate="lag %{x} : %{y:.3f}<extra></extra>"))
    for s in (band, -band):
        fig.add_hline(y=s, line=dict(color=ui.BENCHMARK, width=1, dash="dash"))
    fig.update_layout(title=title, xaxis_title="Retard (semaines)", hovermode="closest", bargap=0.5)
    return fig


def _ic_heatmap(grid: pd.DataFrame, col: str, best: tuple) -> go.Figure:
    m = grid.pivot(index="p", columns="q", values=col)
    rel = m - np.nanmin(m.values)
    fig = go.Figure(go.Heatmap(z=rel.values, x=[f"q = {q}" for q in m.columns], y=[f"p = {p}" for p in m.index],
                               colorscale=list(reversed([[1 - a, c] for a, c in ui.SEQUENTIAL])), xgap=2, ygap=2,
                               text=np.round(rel.values, 1), texttemplate="%{text}",
                               hovertemplate="%{y}, %{x} : Δ" + col + " = %{z:.2f}<extra></extra>",
                               colorbar=dict(thickness=10, outlinewidth=0)))
    fig.update_layout(title=f"Δ{col} relatif au minimum (retenu : ARMA{best})", hovermode="closest",
                      yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False))
    return fig


def _qq(resid: pd.Series) -> go.Figure:
    z = np.sort((resid - resid.mean()) / resid.std())
    q = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q, y=z, mode="markers", marker=dict(size=5, color=ui.STRATEGY, opacity=0.6),
                             name="Résidus standardisés", hovertemplate="%{x:.2f}, %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[q.min(), q.max()], y=[q.min(), q.max()], mode="lines",
                             line=dict(color=ui.INK_2, width=1, dash="dash"), name="Loi normale"))
    fig.update_layout(title="QQ-plot des résidus vs loi normale", xaxis_title="Quantiles théoriques",
                      yaxis_title="Quantiles empiriques", hovermode="closest")
    return fig


def _finding(text: str) -> None:
    ui.note(text, "finding")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _section_arima(md, signals, evaluation, cfg, fp, ticker, split):
    st.markdown("## 1. Modélisation ARIMA des rendements")
    st.markdown(
        "Étude de spécification sur l'échantillon d'apprentissage (du début des données à la date de coupure), "
        "puis évaluation **hors échantillon** des prévisions produites en walk-forward par la pipeline de "
        "production (fenêtre glissante de 5 ans, ordre re-sélectionné chaque année, paramètres ré-estimés "
        "chaque semaine).")
    a = _arima_study(fp, ticker, split, md, cfg.model.arima_max_p, cfg.model.arima_max_q)

    st.markdown("### Ordre d'intégration d — tests ADF et KPSS")
    ui.show_table(a["stationarity"], dec=3)
    _finding(f"Le log-prix de {ticker} n'est pas stationnaire, ses différences premières le sont : "
             f"d = {a['d']}. On modélise donc les log-rendements par un ARMA(p, q), équivalent à un "
             f"ARIMA(p, {a['d']}, q) sur le log-prix. ADF (H0 : racine unitaire) et KPSS (H0 : stationnarité) "
             "ont des hypothèses nulles opposées. Leur concordance rend la conclusion robuste.")

    st.markdown("### Identification — autocorrélations")
    n = len(a["y_train"])
    c1, c2 = st.columns(2)
    with c1:
        ui.chart(_corr_bars(a["acf"], n, "ACF des log-rendements (bande à 95 %)"), 300)
    with c2:
        ui.chart(_corr_bars(a["pacf"], n, "PACF des log-rendements (bande à 95 %)"), 300)

    st.markdown("### Sélection de (p, q) — critères d'information")
    c1, c2 = st.columns(2)
    with c1:
        ui.chart(_ic_heatmap(a["grid"], "BIC", a["bic"]), 330)
    with c2:
        ui.chart(_ic_heatmap(a["grid"], "AICc", a["aicc"]), 330)
    _finding(f"BIC retient ARMA{a['bic']}, AICc retient ARMA{a['aicc']}. La pipeline utilise le BIC. Ce critère "
             "est convergent et pénalise davantage la complexité. Sur des rendements dont l'autocorrélation est "
             "quasi nulle, l'AICc tend à retenir des ARMA(p, p) dont les racines se compensent presque, ce qui "
             "donne des prévisions instables. Un ordre (0, 0) signifie que le meilleur prédicteur linéaire est "
             "la moyenne : c'est un résultat, pas un échec.")
    ui.show_table(a["coef"], dec=4)

    st.markdown("### Diagnostics des résidus")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        ui.show_table(a["diag"], dec=3, index=False)
        ui.chart(_corr_bars(a["resid_acf"], n, "ACF des résidus"), 260)
    with c2:
        ui.chart(_qq(a["resid_train"]), 420)
    diag = a["diag"].set_index("Test")
    arch_p = diag.filter(like="ARCH-LM", axis=0)["p-value"].iloc[0]
    jb_p = diag.loc["Jarque-Bera", "p-value"]
    _finding(f"Ljung-Box : aucune autocorrélation résiduelle significative ne doit subsister. ARCH-LM "
             f"(p = {ui.pvalue(arch_p)}) : {'hétéroscédasticité conditionnelle détectée, ce qui justifie un modèle GARCH sur les résidus' if arch_p < 0.05 else 'pas d’effet ARCH significatif sur cet échantillon'}. "
             f"Jarque-Bera (p = {ui.pvalue(jb_p)}) : {'queues épaisses, ce qui justifie des innovations Student-t' if jb_p < 0.05 else 'normalité non rejetée'}.")

    st.markdown("### Prévisions hors échantillon (walk-forward)")
    s = signals.xs(ticker, level="ticker").dropna(subset=["realized"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.realized, name="Réalisé", line=dict(color=ui.CASH, width=1.2)))
    fig.add_trace(go.Scatter(x=s.index, y=s.arima_mu, name="ARIMA", line=dict(color=ui.STRATEGY, width=2)))
    fig.add_trace(go.Scatter(x=s.index, y=s.hist_mu, name="Moyenne historique", line=dict(color=ui.BENCHMARK, width=1.5, dash="dot")))
    fig.update_layout(title=f"{ticker} — log-rendement sur {cfg.model.horizon} semaines (%) : prévu vs réalisé",
                      yaxis_title="%")
    ui.chart(fig, 340)
    orders = signals.groupby(level="ticker")[["p", "q"]].apply(
        lambda g: g.astype(int).astype(str).agg(",".join, axis=1).value_counts(normalize=True).head(3)
        .rename(lambda x: f"({x})").to_dict())
    ev = evaluation[evaluation.Modèle == "ARIMA"].set_index("Actif")
    ev["Ordres les plus fréquents"] = orders.apply(lambda d: " · ".join(f"{k} {v:.0%}" for k, v in d.items()))
    ui.show_table(ev[["R² OOS vs moyenne hist.", "R² OOS vs zéro", "Taux de bon sens", "IC (Spearman)",
                      "DM stat", "DM p-value", "Ordres les plus fréquents"]], dec=3)
    st.caption("R² OOS (Campbell & Thompson 2008) : 1 − SSE(modèle)/SSE(référence). DM : test de Diebold-Mariano "
               "(correction HLN, variance HAC car les cibles à 4 semaines se chevauchent) contre la moyenne "
               "historique ; une statistique négative indique une erreur plus faible que la référence.")
    ui.note("Constat attendu et assumé : à l'horizon mensuel, la composante linéaire des rendements est quasi "
            "imprévisible (efficience faible des marchés). La valeur d'ARIMA dans la pipeline n'est pas un pouvoir "
            "prédictif élevé, mais une estimation disciplinée de la dérive locale. Ses résidus servent de cible à "
            "la correction non linéaire et de base au modèle de volatilité.")


def _section_xgb(md, signals, importance, evaluation, cfg, fp, ticker):
    st.markdown("## 2. XGBoost sur les résidus ARIMA")
    st.latex(r"\hat r_{t\to t+h} \;=\; \underbrace{\textstyle\sum_{k=1}^{h}\hat r^{\,\mathrm{ARIMA}}_{t+k|t}}_{\text{composante linéaire}}"
             r"\;+\;\underbrace{f_{\mathrm{XGB}}(X_t)}_{\text{correction non linéaire}},\qquad "
             r"f_{\mathrm{XGB}} \approx \mathbb{E}\Big[\textstyle\sum_{k=1}^{h} e_{t+k}\,\Big|\,X_t\Big]")
    st.markdown(
        "Le modèle de boosting apprend la partie des rendements futurs que la structure linéaire n'explique pas, "
        "à partir de 15 variables causales et stationnaires. Seuls les échantillons dont la cible est "
        "entièrement observée à la date de décision sont utilisés (j + h ≤ t). Le modèle est ré-entraîné "
        f"toutes les {cfg.model.xgb_refit_every} semaines sur la fenêtre glissante.")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        doc = pd.DataFrame(FEATURE_DOC, index=["Famille", "Définition", "Rationnel"]).T
        doc.index.name = "Feature"
        st.dataframe(doc, width="stretch", height=420)
    with c2:
        hp = pd.DataFrame({"Valeur": [cfg.model.xgb_n_estimators, cfg.model.xgb_max_depth, cfg.model.xgb_learning_rate,
                                      0.8, 0.8, 40, 20.0, cfg.model.xgb_n_seeds]},
                          index=["Nombre d'arbres", "Profondeur max", "Learning rate", "Subsample (lignes)",
                                 "Colsample (features)", "min_child_weight", "Régularisation L2 (λ)", "Graines (ensemble)"])
        st.dataframe(hp, width="stretch")
        ui.note("Hyper-paramètres fixés a priori et volontairement conservateurs. Avec environ 250 observations "
                "chevauchantes, soit environ 60 observations indépendantes à 4 semaines, toute recherche sur "
                "grille sur-apprendrait la période de validation.")

    st.markdown("### Importance des variables (TreeSHAP)")
    imp = importance.drop(columns=[c for c in importance.columns if c not in FEATURES])
    share = imp.div(imp.sum(axis=1), axis=0)
    asset_share = share.xs(ticker, level="ticker").mean().sort_values()
    c1, c2 = st.columns([1, 1.25])
    with c1:
        fig = go.Figure(go.Bar(x=asset_share.values, y=asset_share.index, orientation="h", marker_color=ui.STRATEGY,
                               hovertemplate="%{y} : %{x:.1%}<extra></extra>"))
        fig.update_layout(title=f"{ticker} — part moyenne de |SHAP| (toutes ré-estimations)",
                          xaxis_tickformat=".0%", hovermode="closest", bargap=0.35)
        ui.chart(fig, 460)
    with c2:
        mat = share.groupby(level="ticker").mean().T.loc[asset_share.index[::-1]]
        fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, colorscale=ui.SEQUENTIAL, xgap=2, ygap=2,
                                   hovertemplate="%{x} · %{y} : %{z:.1%}<extra></extra>",
                                   colorbar=dict(thickness=10, outlinewidth=0, tickformat=".0%")))
        fig.update_layout(title="Part de |SHAP| par actif", hovermode="closest",
                          yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False))
        ui.chart(fig, 460)
    top = asset_share.sort_values(ascending=False).head(5).index.tolist()
    ts = share.xs(ticker, level="ticker")[top].rolling(13, min_periods=1).mean()
    fig = go.Figure()
    for i, f in enumerate(top):
        fig.add_trace(go.Scatter(x=ts.index, y=ts[f], name=f, line=dict(color=ui.CATEGORICAL[i], width=1.8),
                                 hovertemplate=f"{f} : %{{y:.1%}}<extra></extra>"))
    fig.update_layout(title=f"{ticker} — stabilité temporelle des 5 premières variables (moyenne mobile 13 sem.)",
                      yaxis_tickformat=".0%")
    ui.chart(fig, 320)
    _finding(f"Variables les plus influentes pour {ticker} : " + ", ".join(
        f"{f} ({FEATURE_DOC[f][0].lower()}, {asset_share[f]:.0%})" for f in top[:3]) +
        ". L'importance SHAP mesure l'usage que fait le modèle d'une variable, pas sa valeur prédictive "
        "hors échantillon. Cette dernière se lit dans le tableau ci-dessous.")

    st.markdown("### Validation du modèle hybride (hors échantillon)")
    rows = []
    for t, g in signals.groupby(level="ticker"):
        g = g.dropna(subset=["realized"])
        y = g.realized.values
        se_h, se_a = (y - g.hybrid_mu.values) ** 2, (y - g.arima_mu.values) ** 2
        dm, p = diebold_mariano(se_h, se_a, cfg.model.horizon)
        e = evaluation[(evaluation.Actif == t)].set_index("Modèle")
        rows.append({"Actif": t,
                     "R² OOS ARIMA": e.loc["ARIMA", "R² OOS vs moyenne hist."],
                     "R² OOS hybride": e.loc["ARIMA + XGBoost", "R² OOS vs moyenne hist."],
                     "IC hybride": e.loc["ARIMA + XGBoost", "IC (Spearman)"],
                     "Bon sens hybride": e.loc["ARIMA + XGBoost", "Taux de bon sens"],
                     "DM hybride vs ARIMA": dm, "DM p-value": p})
    tbl = pd.DataFrame(rows).set_index("Actif")
    ui.show_table(tbl, pct_cols=["R² OOS ARIMA", "R² OOS hybride", "Bon sens hybride"], dec=3)
    n_sig = int(((tbl["DM p-value"] < 0.05) & (tbl["DM hybride vs ARIMA"] < 0)).sum())
    _finding(f"La correction XGBoost améliore significativement (DM, 5 %) la prévision ARIMA sur {n_sig} actif(s) "
             f"sur {len(tbl)}. IC moyen du modèle hybride : {tbl['IC hybride'].mean():.3f}. Un IC de 0,03 à 0,05 "
             "est économiquement exploitable en allocation (loi fondamentale de Grinold), mais statistiquement "
             "fragile. C'est pourquoi les prévisions n'entrent dans l'optimiseur qu'à travers un rétrécissement "
             "vers un prior d'équilibre (section 4).")


def _section_garch(md, signals, cfg, fp, ticker, split):
    st.markdown("## 3. Volatilité conditionnelle — GARCH")
    st.latex(r"e_t=\sigma_t z_t,\quad z_t\sim t_\nu,\qquad \sigma_t^2=\omega+\alpha\,e_{t-1}^2+\beta\,\sigma_{t-1}^2,"
             r"\qquad \sigma^2_{t\to t+h}=\textstyle\sum_{k=1}^{h}\mathbb{E}_t[\sigma^2_{t+k}]")
    a = _arima_study(fp, ticker, split, md, cfg.model.arima_max_p, cfg.model.arima_max_q)
    g = _garch_study(fp, ticker, split, a["resid_full"])

    st.markdown("### Pré-test et choix de la spécification (échantillon d'apprentissage)")
    pre = a["diag"].set_index("Test")
    pre = pre[pre.index.str.contains("ARCH|résidus²")]
    ui.show_table(pre, dec=3)
    ui.show_table(g["train_specs"], pct_cols=[], dec=2)
    best = g["train_specs"]["BIC"].idxmin() if not g["train_specs"].empty else "—"
    _finding(f"Spécification au BIC minimal sur {ticker} : {best}. La pipeline utilise GARCH(1,1)-t pour tous les "
             "actifs : c'est le modèle le plus parcimonieux, robuste à l'estimation sur 5 ans glissants. Les "
             "innovations Student-t capturent les queues épaisses. L'asymétrie (GJR, EGARCH) n'est significative "
             "que pour les actifs actions et n'apporte qu'un gain marginal de vraisemblance.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Paramètres estimés — GARCH(1,1) Student-t**")
        if g["params"] is not None:
            ui.show_table(g["params"], dec=4)
    with c2:
        st.markdown("**Diagnostics des résidus standardisés z = e / σ**")
        if g["std_diag"] is not None:
            ui.show_table(g["std_diag"], dec=3, index=False)

    oos = g["oos"]
    if oos:
        st.markdown(f"### Évaluation sur l'échantillon de test ({g['n_test']} semaines, paramètres gelés)")
        fv, rv = oos["forecast_var"], oos["realized_var"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rv.index, y=np.sqrt(rv) * np.sqrt(52), name="|e| réalisé (annualisé)",
                             marker_color=ui.CASH, hovertemplate="%{y:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fv.index, y=np.sqrt(fv) * np.sqrt(52), name="σ GARCH(1,1)-t",
                                 line=dict(color=ui.STRATEGY, width=2)))
        ew = oos["candidates"]["EWMA (λ = 0.94)"]
        fig.add_trace(go.Scatter(x=ew.index, y=np.sqrt(ew) * np.sqrt(52), name="σ EWMA",
                                 line=dict(color=ui.BENCHMARK, width=1.4, dash="dot")))
        fig.update_layout(title=f"{ticker} — volatilité prévue à 1 semaine vs choc réalisé (%, annualisé)",
                          bargap=0.1)
        ui.chart(fig, 340)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown("**Fonctions de perte (plus bas = meilleur)**")
            ui.show_table(oos["losses"], dec=3)
        with c2:
            st.markdown("**Diebold-Mariano sur QLIKE (GARCH vs …)**")
            dm = pd.DataFrame(oos["dm"], index=["DM stat", "p-value"]).T
            ui.show_table(dm, dec=3)
        with c3:
            st.markdown("**Régression de Mincer-Zarnowitz**")
            st.dataframe(pd.Series(oos["mz"]).to_frame("Valeur").style.format("{:.3f}"), width="stretch")
        st.caption("QLIKE (Patton 2011) est robuste au bruit du proxy e² ; MZ : e² = a + b·σ̂² + u, une prévision "
                   "efficiente vérifie a = 0 et b = 1 (test de Wald, erreurs HAC).")
        st.markdown("**Ajustement des spécifications sur l'échantillon de test (ré-estimation)**")
        ui.show_table(g["test_specs"], pct_cols=[], dec=2)

    st.markdown("### Volatilité utilisée par l'allocation (walk-forward)")
    s = signals.xs(ticker, level="ticker")
    ann = np.sqrt(52 / cfg.model.horizon) / 100
    realized_vol = (100 * np.log(md.prices[ticker]).diff()).rolling(cfg.model.horizon).std().shift(-cfg.model.horizon) * np.sqrt(52) / 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=realized_vol.index, y=realized_vol, name="Volatilité réalisée (4 sem. suivantes)",
                             line=dict(color=ui.CASH, width=1.2)))
    fig.add_trace(go.Scatter(x=s.index, y=s.sigma_h * ann, name="σ GARCH prévue sur l'horizon",
                             line=dict(color=ui.STRATEGY, width=2)))
    fig.update_layout(title=f"{ticker} — volatilité annualisée prévue vs réalisée", yaxis_tickformat=".0%")
    ui.chart(fig, 320)


def _section_markowitz(md, run, cfg, fp, ticker, signals):
    st.markdown("## 4. Optimisation de Markowitz sous contraintes")
    st.latex(r"\max_{w}\; w^\top(\mu-r_f\mathbf 1)-\frac{\gamma}{2}\,w^\top\Sigma w-c\sum_i|w_i-w_i^{0}|"
             r"\quad\text{s.c.}\quad 0\le w_i\le w_{\max}\,s_i,\;\; \mathbf 1^\top w\le 1,\;\; "
             r"\sqrt{\tfrac{52}{h}\,w^\top\Sigma w}\le\sigma_{\text{cap}}")
    st.latex(r"\mu_i=(1-\kappa)\,\underbrace{\big(r_f+\mathrm{SR}_{\text{prior}}\sigma_i\big)}_{\text{prior d'équilibre}}"
             r"+\kappa\,\underbrace{\big(e^{\hat r_i^{\,\mathrm{ARIMA+XGB}}+\sigma_i^2/2}-1\big)}_{\text{vue du modèle}},"
             r"\qquad \Sigma=D_{\mathrm{GARCH}}\,R_{\mathrm{LW}}\,D_{\mathrm{GARCH}}")
    p = cfg.portfolio
    st.markdown(
        f"- **Fonction objectif** : utilité moyenne-variance (γ = {p.risk_aversion:g}) nette des coûts de "
        "transaction. Ce n'est pas le ratio de Sharpe : avec un actif sans risque, le Sharpe est invariant "
        "d'échelle (w et λw ont le même Sharpe). Il fixe la direction du portefeuille tangent, pas le montant "
        "investi. Hors contraintes actives, la solution w* = Σ⁻¹(μ − r_f)/γ est colinéaire au portefeuille de "
        "Sharpe maximal.\n"
        f"- **Intégration des prédictions** : les prévisions ARIMA-XGBoost entrent dans μ avec un poids "
        f"κ = {p.view_weight:g}, par rétrécissement vers un prior « Sharpe constant » (SR = {p.prior_sharpe:g}), "
        "dans l'esprit de Black-Litterman. Avec κ = 0, l'allocation ne dépend que du risque.\n"
        "- **Covariance** : volatilités GARCH prévues sur l'horizon, corrélations sur 2 ans rétrécies par "
        "Ledoit-Wolf, ce qui garantit une matrice bien conditionnée.\n"
        f"- **Safety-First** : long-only, pas de levier, poche de T-Bills endogène, volatilité ex-ante plafonnée "
        f"à {p.vol_cap:.0%}, poids maximal de {p.max_weight:.0%} modulé par le filtre de tendance.")

    model = run.model
    dates = model.dates[model.dates >= run.strategy.equity.index[0] - pd.Timedelta(weeks=1)]
    date = st.select_slider("Date de rééquilibrage analysée", options=list(dates), value=dates[-1],
                            format_func=lambda d: f"{d:%d/%m/%Y}", key="mk_date")
    x = model.inputs(date)
    w = model.weights(date, None, cfg.backtest.cost_bps / 1e4)
    ann = 52 / cfg.model.horizon
    tbl = pd.DataFrame({
        "Score de tendance": x.trend, "Borne haute": x.upper,
        "Volatilité (horizon)": x.sigma * np.sqrt(ann),
        "Prior": x.prior * ann, "Prévision": x.forecast * ann, "μ retenu": x.mu * ann, "Poids": w})
    tbl.loc["T-Bills (cash)"] = [np.nan, np.nan, 0.0, x.rf_h * ann, x.rf_h * ann, x.rf_h * ann, max(0.0, 1 - w.sum())]
    tbl = tbl.astype(float)
    c1, c2 = st.columns([1.25, 1])
    with c1:
        ui.show_table(tbl, dec=2)
        st.caption("Rendements et volatilités annualisés (×52/h) pour la lisibilité ; l'optimisation est menée "
                   "sur l'horizon de 4 semaines.")
    with c2:
        fr_c = model.frontier(date, constrained=True)
        fr_u = model.frontier(date, constrained=False)
        fig = go.Figure()
        if not fr_u.empty:
            fig.add_trace(go.Scatter(x=fr_u.vol, y=fr_u.ret, name="Frontière long-only", mode="lines",
                                     line=dict(color=ui.INK_3, width=1.5, dash="dot")))
        if not fr_c.empty:
            fig.add_trace(go.Scatter(x=fr_c.vol, y=fr_c.ret, name="Frontière sous filtre SMA", mode="lines",
                                     line=dict(color=ui.STRATEGY, width=2)))
        vol_a = x.sigma * np.sqrt(ann)
        fig.add_trace(go.Scatter(x=vol_a, y=x.mu * ann, mode="markers+text", text=list(x.mu.index),
                                 textposition="top center", textfont=dict(size=10, color=ui.INK_2),
                                 marker=dict(size=8, color=ui.CASH, line=dict(color=ui.INK_2, width=1)),
                                 name="Actifs", hovertemplate="%{text} : σ %{x:.1%}, μ %{y:.1%}<extra></extra>"))
        pv = np.sqrt(w.values @ x.cov.values @ w.values * ann)
        pr = (w.values @ x.mu.values + (1 - w.sum()) * x.rf_h) * ann
        fig.add_trace(go.Scatter(x=[pv], y=[pr], mode="markers", name="Portefeuille retenu",
                                 marker=dict(size=13, symbol="diamond", color=ui.BENCHMARK, line=dict(color="white", width=2)),
                                 hovertemplate="σ %{x:.1%}, μ %{y:.1%}<extra></extra>"))
        fig.add_vline(x=p.vol_cap, line=dict(color=ui.INK_3, width=1, dash="dash"),
                      annotation_text="plafond de volatilité", annotation_font=dict(size=10, color=ui.INK_2))
        fig.update_layout(title=f"Frontière efficiente au {date:%d/%m/%Y}", xaxis_title="Volatilité annualisée",
                          yaxis_title="Rendement espéré annualisé", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          hovermode="closest")
        ui.chart(fig, 460, legend_bottom=True)

    st.markdown("### Contrôle des bornes par les moyennes mobiles long terme")
    st.markdown(
        f"Score de tendance s = moyenne de 1{{P > SMA_L}} pour L ∈ {set(cfg.model.sma_windows)} semaines "
        "(la SMA 40 semaines correspond à la règle 10 mois / 200 jours de Faber, 2007). La borne haute d'un actif vaut "
        "w_max · s. Le filtre ne force jamais l'achat. Il retire progressivement l'actif de l'ensemble admissible "
        "quand sa tendance se dégrade, et le capital libéré va en T-Bills ou vers les actifs en tendance. "
        "Moyenner trois horizons réduit la dépendance à un paramètre unique et les allers-retours.")
    price = md.prices[ticker]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price, name=ticker, line=dict(color=ui.INK, width=1.6)))
    for i, L in enumerate(cfg.model.sma_windows):
        fig.add_trace(go.Scatter(x=price.index, y=sma(price, L), name=f"SMA {L} sem.",
                                 line=dict(color=[ui.STRATEGY, ui.BENCHMARK, ui.EQUAL_WEIGHT][i % 3], width=1.4)))
    fig.update_layout(title=f"{ticker} — prix ajusté et moyennes mobiles", yaxis_type="log")
    ui.chart(fig, 320)
    score = model.trend[ticker]
    fig = go.Figure(go.Scatter(x=score.index, y=score, line=dict(color=ui.STRATEGY, width=1.5, shape="hv"),
                               fill="tozeroy", fillcolor="rgba(42,120,214,0.12)", name="Score",
                               hovertemplate="%{y:.2f}<extra></extra>"))
    fig.update_layout(title=f"{ticker} — score de tendance s (borne haute = {p.max_weight:.0%} × s)",
                      yaxis=dict(range=[-0.05, 1.05], tickvals=[0, 1 / 3, 2 / 3, 1], ticktext=["0", "1/3", "2/3", "1"]))
    ui.chart(fig, 220)
    tstats = _trend_stats(fp, tuple(cfg.model.sma_windows), cfg.model.horizon, md)
    ui.show_table(tstats, pct_cols=["Temps en tendance haussière", "Rdt ann. | haussier", "Rdt ann. | baissier",
                                    "Vol ann. | haussier", "Vol ann. | baissier"], dec=2)
    _finding("Justification statistique du filtre : comparaison des rendements à 4 semaines (non chevauchants) "
             "conditionnellement au signal. Le test de Welch porte sur la différence de moyennes, le test de Levene "
             "sur l'égalité des variances. L'apport principal attendu d'un filtre de tendance est la réduction de "
             "la volatilité et des pertes extrêmes en régime baissier, plus qu'un écart de rendement moyen "
             "significatif (Moskowitz, Ooi & Pedersen 2012 ; Hurst, Ooi & Pedersen 2017).")

    st.markdown("### Contribution de chaque composante (ablation)")
    abl = _ablation(fp, f"{cfg.portfolio.key()}{cfg.backtest.key()}", md, signals, cfg)
    ui.show_table(abl[["Rendement annualisé", "Volatilité annualisée", "Ratio de Sharpe", "Max Drawdown",
                       "Ratio de Calmar", "Turnover annuel", "Exposition moyenne"]], dec=2)
    ui.note("Chaque ligne retire une seule brique, toutes choses égales par ailleurs, sur la même période, avec les "
            "mêmes coûts et le même tranching. C'est le test le plus direct de la valeur ajoutée de chaque étape. "
            "Une brique qui n'améliore pas le profil rendement/risque net de frais est candidate à la suppression.")


def _section_execution(md, run, cfg, fp, signals):
    st.markdown("## 5. Timing, tranching et exécution")
    b = cfg.backtest
    st.markdown(
        f"- **Données hebdomadaires** (clôture du vendredi, prix ajustés) : elles atténuent le bruit de "
        "microstructure et les effets jour de la semaine des données quotidiennes, tout en conservant environ "
        "1 000 observations pour l'estimation.\n"
        f"- **Rééquilibrage toutes les {b.rebalance_every} semaines**, aligné sur l'horizon de prévision "
        f"(h = {cfg.model.horizon}) : chaque décision est évaluée sur la période pour laquelle elle a été prévue.\n"
        f"- **Décalage d'exécution de {b.execution_lag} semaine(s)** : le signal calculé à la clôture de s "
        "est exécuté à la clôture suivante. Aucun ordre n'est passé au prix qui a servi à le calculer.\n"
        f"- **Tranching** : le capital est réparti en {b.n_tranches} sous-portefeuilles qui se rééquilibrent à "
        "tour de rôle, une tranche chaque semaine. Le résultat ne dépend plus du jour arbitraire de "
        "rééquilibrage (le « timing luck » de Hoffstein, Faber & Braun, 2020), et la transition entre "
        "allocations est lissée.\n"
        f"- **Coûts** : {b.cost_bps:g} bps par unité de volume échangé (commission et demi-spread d'ETF liquides), "
        "intégrés à la fois dans la fonction objectif et dans la comptabilité.\n"
        f"- **Buffer de rééquilibrage** : si le turnover one-way proposé ½Σ|w* − w| est inférieur à {b.buffer:.1%}, "
        "la tranche conserve ses positions. Exception : une position qui dépasse sa nouvelle borne de tendance "
        "est toujours ramenée sous la borne, car les sorties de risque ne sont jamais différées.")

    res = run.strategy
    fig = go.Figure()
    for i, col in enumerate(res.tranche_equity.columns):
        eq = res.tranche_equity[col] / res.tranche_equity[col].iloc[0]
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=col, line=dict(color=ui.CATEGORICAL[(i + 3) % 8], width=1.2)))
    fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, name="Agrégé", line=dict(color=ui.INK, width=2.2)))
    fig.update_layout(title="Valeur de chaque tranche (base 1) et portefeuille agrégé", yaxis_type="log")
    ui.chart(fig, 360)
    disp = tranche_dispersion(res, md.rf)
    ui.show_table(disp[["Rendement annualisé", "Volatilité annualisée", "Ratio de Sharpe", "Max Drawdown", "Ratio de Calmar"]])
    spread = disp["Ratio de Sharpe"].iloc[:-1]
    _finding(f"Dispersion du Sharpe entre tranches : {spread.min():.2f} à {spread.max():.2f}. "
             "Cet écart est la part du résultat due au seul choix de la semaine de rééquilibrage. Une stratégie "
             "évaluée sur une seule tranche en hérite intégralement.")

    years = len(res.returns) / 52
    ui.kpi_row([
        ("Turnover annuel (one-way)", ui.pct(res.turnover.sum() / years, 0), None),
        ("Coûts annuels", ui.pct(res.costs.sum() / years, 2), f"{b.cost_bps:g} bps par transaction"),
        ("Rééquilibrages exécutés", str(res.n_trades), None),
        ("Évités par le buffer", str(res.n_skipped), f"{res.n_skipped / max(res.n_trades + res.n_skipped, 1):.0%} des décisions"),
    ])
    st.write("")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.turnover.index, y=res.turnover.rolling(52).sum(), name="Turnover 52 sem.",
                             line=dict(color=ui.STRATEGY, width=1.8)))
    fig.update_layout(title="Turnover one-way glissant sur 52 semaines", yaxis_tickformat=".0%")
    ui.chart(fig, 280)

    st.markdown("### Sensibilité au buffer et aux coûts")
    if st.button("Lancer l'analyse de sensibilité (12 backtests)", key="run_sens"):
        st.session_state["sens_on"] = True
    if st.session_state.get("sens_on"):
        sens = _sensitivity(fp, f"{cfg.portfolio.key()}{cfg.backtest.key()}", md, signals, cfg)
        ui.show_table(sens.set_index(["Coût (bps)", "Buffer"]), pct_cols=["Coûts annuels"], dec=2)
        st.caption("Lecture : le buffer réduit mécaniquement le turnover et les coûts. Il est utile tant que "
                   "le Sharpe net ne se dégrade pas, c'est-à-dire tant que les transactions évitées étaient du "
                   "bruit et non du signal.")


def render(md: MarketData, signals: pd.DataFrame, importance: pd.DataFrame, run, cfg: Config, fp: str) -> None:
    tickers = list(md.prices.columns)
    c1, c2, _ = st.columns([1, 2, 2])
    with c1:
        ticker = st.selectbox("Actif étudié", tickers, key="r_ticker")
    with c2:
        dates = md.prices.index
        default = dates[int(len(dates) * 0.7)]
        split = st.select_slider("Coupure apprentissage / test (études ARIMA et GARCH)",
                                 options=list(dates[int(len(dates) * 0.4): int(len(dates) * 0.9)]),
                                 value=default, format_func=lambda d: f"{d:%m/%Y}", key="r_split")
    evaluation = _forecast_eval(fp, cfg.model.key(), signals, cfg.model.horizon)
    t1, t2, t3, t4, t5 = st.tabs(["ARIMA", "XGBoost sur résidus", "GARCH", "Markowitz et tendance", "Timing et exécution"])
    with t1:
        _section_arima(md, signals, evaluation, cfg, fp, ticker, split)
    with t2:
        _section_xgb(md, signals, importance, evaluation, cfg, fp, ticker)
    with t3:
        _section_garch(md, signals, cfg, fp, ticker, split)
    with t4:
        _section_markowitz(md, run, cfg, fp, ticker, signals)
    with t5:
        _section_execution(md, run, cfg, fp, signals)
