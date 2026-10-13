"""
Analyse de sensibilité : balayage des paramètres de portefeuille et d'exécution
sur le panel de signaux walk-forward mis en cache.

Deux études :
- « un paramètre à la fois » (OAT) : chaque paramètre varie seul, les autres
  restant à leur valeur par défaut (src/config.py) ;
- grille factorielle complète γ × κ × plafond de volatilité × poids maximal ×
  filtre de tendance (5 × 5 × 5 × 4 × 2 = 1 000 backtests).

Les signaux ne dépendent pas de ces paramètres : ils sont calculés une seule
fois, seuls l'optimisation et le backtest sont rejoués.

Usage :
    python -m scripts.parameter_sweep                    # écrit docs/results/parameter_sweep.csv
    python -m scripts.parameter_sweep --jobs 4 --out chemin.csv
"""
from __future__ import annotations

import argparse
import itertools
import time
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from src.config import Config
from src.data import load_market_data
from src.metrics import performance, relative
from src.pipeline import run_strategy
from src.signals import compute_signal_panel

warnings.filterwarnings("ignore")

PORTFOLIO_KEYS = ("risk_aversion", "view_weight", "vol_cap", "max_weight", "use_trend_filter", "forecast_source")
BACKTEST_KEYS = ("start", "cost_bps", "buffer", "n_tranches")

OAT_GRID = {
    "risk_aversion": [2, 3, 4, 6, 8, 10, 15, 20],
    "view_weight": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    "vol_cap": [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20],
    "max_weight": [0.15, 0.20, 0.30, 0.40, 0.50, 0.60],
    "use_trend_filter": [True, False],
    "start": ["2011-01-01", "2013-01-01", "2015-01-01", "2018-01-01", "2020-01-01", "2022-01-01"],
    "cost_bps": [0, 5, 10, 20, 30, 40],
    "buffer": [0, 0.025, 0.05, 0.075, 0.10, 0.15],
    "n_tranches": [1, 2, 4],
    "forecast_source": ["hybrid_mu", "arima_mu", "hist_mu"],
}
FACTORIAL = {
    "risk_aversion": [2, 4, 6, 10, 20],
    "view_weight": [0, 0.25, 0.5, 0.75, 1.0],
    "vol_cap": [0.06, 0.08, 0.10, 0.12, 0.15],
    "max_weight": [0.2, 0.3, 0.45, 0.6],
    "use_trend_filter": [True, False],
}


def defaults(cfg: Config) -> dict:
    p, b = cfg.portfolio, cfg.backtest
    return {**{k: getattr(p, k) for k in PORTFOLIO_KEYS}, **{k: getattr(b, k) for k in BACKTEST_KEYS}}


def evaluate(md, signals, base: Config, params: dict) -> dict:
    cfg = replace(base,
                  portfolio=replace(base.portfolio, **{k: params[k] for k in PORTFOLIO_KEYS}),
                  backtest=replace(base.backtest, **{k: params[k] for k in BACKTEST_KEYS}))
    run = run_strategy(md, signals, cfg)
    s = run.strategy
    years = len(s.returns) / 52
    out = performance(s.returns, md.rf)
    out.update({f"rel_{k}": v for k, v in relative(s.returns, run.benchmark.returns).items()})
    mb, me = performance(run.benchmark.returns, md.rf), performance(run.equal_weight.returns, md.rf)
    out.update({
        "bench_sharpe": mb["Ratio de Sharpe"], "bench_cagr": mb["Rendement annualisé"],
        "bench_vol": mb["Volatilité annualisée"], "bench_mdd": mb["Max Drawdown"],
        "ew_sharpe": me["Ratio de Sharpe"], "ew_cagr": me["Rendement annualisé"], "ew_mdd": me["Max Drawdown"],
        "turnover": s.turnover.sum() / years, "costs": s.costs.sum() / years,
        "exposure": s.weights.sum(axis=1).mean(), "n_trades": s.n_trades, "n_skipped": s.n_skipped,
    })
    out.update(params)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--out", default="docs/results/parameter_sweep.csv")
    args = parser.parse_args()

    base = Config()
    md = load_market_data()
    signals, _ = compute_signal_panel(md, base.model, n_jobs=args.jobs)
    d = defaults(base)

    jobs = [("oat_" + k, {**d, k: v}) for k, values in OAT_GRID.items() for v in values]
    for combo in itertools.product(*FACTORIAL.values()):
        jobs.append(("grid", {**d, **dict(zip(FACTORIAL, combo))}))

    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(evaluate)(md, signals, base, p) for _, p in jobs)
    df = pd.DataFrame(rows)
    df.insert(0, "study", [s for s, _ in jobs])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} backtests en {time.time() - t0:.0f} s → {args.out}")


if __name__ == "__main__":
    main()
