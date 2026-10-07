"""
Safety-First Multi-Asset — terminal de recherche et de suivi.

Lancement :  streamlit run app.py

Organisation :
  1. Univers & diversification   statistiques descriptives, corrélations, clustering
  2. Recherche & preuves          ARIMA → XGBoost → GARCH → Markowitz/SMA → exécution
  3. Backtest & performance       résultats walk-forward nets de frais vs benchmark

Les signaux walk-forward (le calcul coûteux) sont produits une seule fois puis
mis en cache sur disque ; tous les paramètres de portefeuille et d'exécution de
la barre latérale sont ensuite ré-évalués en quelques secondes.
"""
from __future__ import annotations

import warnings
from dataclasses import replace

import pandas as pd
import streamlit as st

from src import ui
from src.config import BENCHMARK_NAME, TICKERS, BacktestConfig, Config, PortfolioConfig
from src.data import load_market_data
from src.pipeline import run_strategy
from src.signals import cache_paths, compute_signal_panel, data_fingerprint
from src.views import performance, research, universe

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Safety-First Multi-Asset", page_icon="▪", layout="wide",
                   initial_sidebar_state="expanded")
ui.register_template()
ui.inject_css()


# ---------------------------------------------------------------------------
# Chargement (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des données de marché…")
def get_data(refresh_token: int):
    return load_market_data(refresh=refresh_token > 0)


@st.cache_resource(show_spinner=False)
def get_signals(fp: str, model_key: str, _md, _cfg):
    bar = st.progress(0.0, text="Estimation walk-forward ARIMA · GARCH · XGBoost (une fois, puis cache disque)…")
    out = compute_signal_panel(_md, _cfg.model, progress=lambda f: bar.progress(f, text=f"Signaux walk-forward : {f:.0%} des actifs"))
    bar.empty()
    return out


@st.cache_resource(show_spinner="Backtest walk-forward…")
def get_run(fp: str, pcfg: PortfolioConfig, bcfg: BacktestConfig, model_key: str, _md, _signals):
    return run_strategy(_md, _signals, Config(portfolio=pcfg, backtest=bcfg))


# ---------------------------------------------------------------------------
# Barre latérale : paramètres
# ---------------------------------------------------------------------------
defaults = Config()
with st.sidebar:
    st.markdown("## Données")
    if "refresh" not in st.session_state:
        st.session_state.refresh = 0
    if st.button("Actualiser depuis Yahoo Finance", width="stretch"):
        st.session_state.refresh += 1
        get_data.clear()

    st.markdown("## Portefeuille")
    gamma = st.slider("Aversion au risque γ", 2.0, 20.0, defaults.portfolio.risk_aversion, 0.5)
    kappa = st.slider("Poids des prévisions κ", 0.0, 1.0, defaults.portfolio.view_weight, 0.05,
                      help="0 : allocation fondée uniquement sur le risque (prior). 1 : prévisions ARIMA-XGBoost seules.")
    vol_cap = st.slider("Plafond de volatilité ex-ante", 0.04, 0.20, defaults.portfolio.vol_cap, 0.01, format="%.2f")
    w_max = st.slider("Poids maximal par actif", 0.15, 0.60, defaults.portfolio.max_weight, 0.05, format="%.2f")
    trend_on = st.toggle("Filtre de tendance SMA", value=defaults.portfolio.use_trend_filter)

    st.markdown("## Exécution")
    start = st.date_input("Début du backtest", value=pd.Timestamp(defaults.backtest.start))
    cost = st.slider("Coût de transaction (bps, one-way)", 0.0, 40.0, defaults.backtest.cost_bps, 1.0)
    buffer = st.slider("Buffer de rééquilibrage (turnover one-way)", 0.0, 0.15, defaults.backtest.buffer, 0.005, format="%.3f")
    tranches = st.select_slider("Nombre de tranches", options=[1, 2, 4], value=defaults.backtest.n_tranches)

    st.markdown("## Modèle")
    m = defaults.model
    st.caption(f"Horizon {m.horizon} sem. · fenêtre {m.train_window} sem. · ARMA(p≤{m.arima_max_p}, q≤{m.arima_max_q}) "
               f"au {m.arima_criterion} · GARCH(1,1)-t · XGBoost {m.xgb_n_estimators} arbres, profondeur "
               f"{m.xgb_max_depth}, calibré hors échantillon · SMA {', '.join(map(str, m.sma_windows))} sem.")

cfg = Config(
    model=defaults.model,
    portfolio=replace(defaults.portfolio, risk_aversion=gamma, view_weight=kappa, vol_cap=vol_cap,
                      max_weight=w_max, use_trend_filter=trend_on),
    backtest=replace(defaults.backtest, start=str(pd.Timestamp(start).date()), cost_bps=cost, buffer=buffer,
                     n_tranches=tranches),
)

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
try:
    md = get_data(st.session_state.refresh)
except Exception as exc:  # pas de cache local et pas d'accès réseau
    st.title("Safety-First Multi-Asset")
    st.error(f"Données indisponibles : {exc}. Vérifiez l'accès à Yahoo Finance puis relancez.")
    st.stop()

fp = data_fingerprint(md)
st.markdown('<div class="eyebrow">Recherche quantitative · Allocation tactique multi-actifs</div>', unsafe_allow_html=True)
st.title("Safety-First Multi-Asset")
st.markdown(
    f'<div class="subtitle">{len(TICKERS)} ETF · données hebdomadaires du {md.prices.index[0]:%d/%m/%Y} au '
    f'{md.prices.index[-1]:%d/%m/%Y} · ARIMA-XGBoost · GARCH · Markowitz sous filtre SMA · '
    f'benchmark {BENCHMARK_NAME}</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Univers & diversification", "Recherche & preuves statistiques", "Backtest & performance"])

with tab1:
    universe.render(md, fp)

signals_ready = all(p.exists() for p in cache_paths(md, cfg.model))
if not signals_ready and not st.session_state.get("compute_signals"):
    msg = ("Les signaux walk-forward ne sont pas encore calculés pour ce jeu de données. Le calcul ré-estime "
           "chaque semaine ARIMA, GARCH et XGBoost pour chaque actif (environ 5 à 10 minutes sur 4 cœurs), puis "
           "il est mis en cache sur disque. Il peut aussi être lancé hors interface : python -m scripts.run_research")
    for tab in (tab2, tab3):
        with tab:
            ui.note(msg)
            if st.button("Calculer les signaux", key=f"go_{id(tab)}", type="primary"):
                st.session_state.compute_signals = True
                st.rerun()
    st.stop()

with tab2:
    signals, importance = get_signals(fp, cfg.model.key(), md, cfg)
run = get_run(fp, cfg.portfolio, cfg.backtest, cfg.model.key(), md, signals)

with tab2:
    research.render(md, signals, importance, run, cfg, fp)
with tab3:
    performance.render(md, run, cfg)
