"""
Orchestration : signaux walk-forward → allocation → backtests (stratégie et
benchmarks) → variantes d'ablation. Point d'entrée unique de l'interface et
du script de recherche.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.backtest import BacktestResult, model_target, run_backtest, static_target
from src.config import BENCHMARK_WEIGHTS, Config
from src.data import MarketData
from src.metrics import performance
from src.portfolio import AllocationModel


@dataclass
class StrategyRun:
    strategy: BacktestResult
    benchmark: BacktestResult | None
    equal_weight: BacktestResult | None
    model: AllocationModel


def run_strategy(md: MarketData, signals: pd.DataFrame, cfg: Config, progress=None,
                 model: AllocationModel | None = None, with_benchmarks: bool = True) -> StrategyRun:
    rets = md.returns
    bt = cfg.backtest
    model = model or AllocationModel(md.prices, md.rf, signals, cfg.model, cfg.portfolio)
    sig_dates = model.dates
    strat = run_backtest(rets, md.rf, model_target(model, bt.cost_bps / 1e4), bt,
                         start=pd.Timestamp(bt.start), signal_dates=sig_dates, progress=progress)
    if not with_benchmarks:
        return StrategyRun(strat, None, None, model)
    start = strat.equity.index[0]
    # Les benchmarks démarrent à la même date d'exécution que la stratégie.
    bench_cfg = replace(bt, buffer=0.0, execution_lag=0)
    bench = run_backtest(rets, md.rf, static_target(BENCHMARK_WEIGHTS), bench_cfg, start=start)
    n = len(md.prices.columns)
    ew = run_backtest(rets, md.rf, static_target({t: 1 / n for t in md.prices.columns}), bench_cfg, start=start)
    return StrategyRun(strat, bench, ew, model)


def ablation(md: MarketData, signals: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Contribution de chaque brique : on retire une composante à la fois."""
    p = cfg.portfolio
    variants = {
        "Stratégie complète": p,
        "Sans prévisions (κ = 0, prior seul)": replace(p, view_weight=0.0),
        "ARIMA seul (sans XGBoost)": replace(p, forecast_source="arima_mu"),
        "Sans filtre de tendance SMA": replace(p, use_trend_filter=False),
    }
    rows = {}
    for name, pc in variants.items():
        res = run_strategy(md, signals, replace(cfg, portfolio=pc), with_benchmarks=False).strategy
        m = performance(res.returns, md.rf)
        m["Turnover annuel"] = res.turnover.sum() / (len(res.turnover) / 52)
        m["Exposition moyenne"] = res.weights.sum(axis=1).mean()
        rows[name] = m
    return pd.DataFrame(rows).T


def sensitivity(md: MarketData, signals: pd.DataFrame, cfg: Config,
                buffers=(0.0, 0.025, 0.05, 0.10), costs=(5.0, 10.0, 20.0)) -> pd.DataFrame:
    """Sensibilité au buffer de rééquilibrage et au niveau de coûts."""
    rows = []
    model = AllocationModel(md.prices, md.rf, signals, cfg.model, cfg.portfolio)
    for c in costs:
        for b in buffers:
            bt = replace(cfg.backtest, buffer=b, cost_bps=c)
            res = run_strategy(md, signals, replace(cfg, backtest=bt), model=model,
                               with_benchmarks=False).strategy
            m = performance(res.returns, md.rf)
            years = len(res.returns) / 52
            rows.append({"Coût (bps)": c, "Buffer": b,
                         "Rendement annualisé": m["Rendement annualisé"],
                         "Ratio de Sharpe": m["Ratio de Sharpe"],
                         "Turnover annuel": res.turnover.sum() / years,
                         "Coûts annuels": res.costs.sum() / years,
                         "Transactions": res.n_trades, "Rééquilibrages évités": res.n_skipped})
    return pd.DataFrame(rows)


def tranche_dispersion(res: BacktestResult, rf: pd.Series) -> pd.DataFrame:
    """Métriques de chaque tranche prise isolément : mesure du risque de timing."""
    rows = {}
    for col in res.tranche_equity.columns:
        r = res.tranche_equity[col].pct_change().fillna(0)
        rows[col] = performance(r, rf)
    rows["Agrégé (tranching)"] = performance(res.returns, rf)
    return pd.DataFrame(rows).T


def forecast_evaluation(signals: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Évaluation hors échantillon (walk-forward) des prévisions, par actif."""
    from src.stats_tests import diebold_mariano, hit_rate, information_coefficient, oos_r2

    rows = []
    for t, g in signals.groupby(level="ticker"):
        g = g.dropna(subset=["realized"])
        y = g.realized.values
        for name, col in (("Moyenne historique", "hist_mu"), ("ARIMA", "arima_mu"), ("ARIMA + XGBoost", "hybrid_mu")):
            f = g[col].values
            dm, p = diebold_mariano((y - f) ** 2, (y - g.hist_mu.values) ** 2, horizon) if col != "hist_mu" else (np.nan, np.nan)
            rows.append({"Actif": t, "Modèle": name,
                         "RMSE (%)": float(np.sqrt(np.mean((y - f) ** 2))),
                         "R² OOS vs moyenne hist.": oos_r2(y, f, g.hist_mu.values),
                         "R² OOS vs zéro": oos_r2(y, f, np.zeros_like(y)),
                         "Taux de bon sens": hit_rate(y, f),
                         "IC (Spearman)": information_coefficient(y, f),
                         "DM stat": dm, "DM p-value": p, "N": len(y)})
    return pd.DataFrame(rows)
