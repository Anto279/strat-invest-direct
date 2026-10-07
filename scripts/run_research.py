"""
Pipeline de recherche hors interface : téléchargement des données, calcul des
signaux walk-forward (mis en cache pour l'application), backtest et rapport.

Usage :
    python -m scripts.run_research            # utilise le cache local s'il existe
    python -m scripts.run_research --refresh  # force le re-téléchargement
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

from src.config import BENCHMARK_NAME, EQUAL_WEIGHT_NAME, Config
from src.data import load_market_data
from src.metrics import performance
from src.pipeline import ablation, forecast_evaluation, run_strategy
from src.signals import compute_signal_panel

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.3f}".format)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-télécharger les données")
    parser.add_argument("--jobs", type=int, default=-1, help="processus parallèles (défaut : tous les cœurs)")
    args = parser.parse_args()

    cfg = Config()
    md = load_market_data(refresh=args.refresh)
    print(f"Données : {md.prices.index[0]:%Y-%m-%d} → {md.prices.index[-1]:%Y-%m-%d}, {md.prices.shape[1]} actifs")

    signals, _ = compute_signal_panel(md, cfg.model, n_jobs=args.jobs,
                                      progress=lambda f: print(f"  signaux : {f:.0%}", flush=True))
    run = run_strategy(md, signals, cfg)
    perf = pd.DataFrame({
        "Stratégie": performance(run.strategy.returns, md.rf),
        BENCHMARK_NAME: performance(run.benchmark.returns, md.rf),
        EQUAL_WEIGHT_NAME: performance(run.equal_weight.returns, md.rf),
    }).T
    print("\nPerformance hors échantillon\n", perf)

    ev = forecast_evaluation(signals, cfg.model.horizon)
    print("\nÉvaluation des prévisions (moyenne sur les actifs)\n",
          ev.groupby("Modèle")[["R² OOS vs moyenne hist.", "IC (Spearman)", "Taux de bon sens"]].mean())
    print("\nAblation\n", ablation(md, signals, cfg)[["Rendement annualisé", "Volatilité annualisée",
                                                      "Ratio de Sharpe", "Max Drawdown", "Ratio de Calmar"]])


if __name__ == "__main__":
    main()
