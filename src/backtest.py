"""
Moteur de backtest hebdomadaire avec tranching, coûts et bande de non-transaction.

Comptabilité (pas hebdomadaire, rendements SIMPLES) :
    V_{t} = V_{t-1} · [ Σ_i w_{i,t-1} (1 + R_{i,t}) + (1 − Σ_i w_{i,t-1}) (1 + r_{f,t-1}) ]
    w_{i,t} = w_{i,t-1} (1 + R_{i,t}) · V_{t-1} / V_t          (dérive des poids)
Le rendement d'un portefeuille est une moyenne pondérée de rendements
simples, jamais de log-rendements (erreur de la version précédente).

Calendrier :
- signal calculé à la clôture de la semaine s (information ≤ s) ;
- exécution à la clôture de s + lag (lag = 1 par défaut : aucune exécution au
  prix ayant servi à calculer le signal) ;
- chaque tranche k ∈ {0..K−1} se rééquilibre toutes les R semaines avec un
  décalage de k·R/K semaines. Le portefeuille agrégé est la somme des
  tranches : il neutralise le « timing luck » (Hoffstein, Faber & Braun 2020)
  lié au choix arbitraire du jour de rééquilibrage.

Coûts et buffer :
- coût proportionnel c (bps, one-way) appliqué au volume échangé ;
- si le turnover one-way proposé ½ Σ|w* − w| est inférieur au buffer, la
  transaction est annulée — SAUF pour les positions qui dépassent leur
  nouvelle borne de tendance : les sorties de risque sont toujours exécutées.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.config import BacktestConfig

TargetFn = Callable[[pd.Timestamp, pd.Series], tuple[pd.Series, pd.Series]]


@dataclass
class BacktestResult:
    equity: pd.Series            # valeur agrégée (base 1)
    returns: pd.Series           # rendements hebdomadaires nets
    weights: pd.DataFrame        # poids agrégés (fin de semaine, après exécution)
    tranche_equity: pd.DataFrame
    turnover: pd.Series          # turnover one-way agrégé, par semaine
    costs: pd.Series             # coûts en fraction de la valeur agrégée
    n_trades: int
    n_skipped: int

    @property
    def cash(self) -> pd.Series:
        return (1 - self.weights.sum(axis=1)).clip(lower=0)


def run_backtest(returns: pd.DataFrame, rf: pd.Series, target_fn: TargetFn, cfg: BacktestConfig,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                 signal_dates: pd.DatetimeIndex | None = None, progress=None) -> BacktestResult:
    tickers = list(returns.columns)
    dates = returns.index
    if signal_dates is not None:
        first_signal = signal_dates.min()
        lo = max(pd.Timestamp(start or dates[0]), first_signal)
    else:
        lo = pd.Timestamp(start or dates[0])
    p0 = int(dates.searchsorted(lo)) + cfg.execution_lag
    p_end = len(dates) if end is None else int(dates.searchsorted(pd.Timestamp(end), side="right"))
    if p0 >= p_end:
        raise ValueError("Période de backtest vide.")

    K, R, c = cfg.n_tranches, cfg.rebalance_every, cfg.cost_bps / 1e4
    offsets = [k * R // K for k in range(K)]
    V = np.full(K, 1.0 / K)
    W = np.zeros((K, len(tickers)))
    R_arr, rf_arr = returns.values, rf.reindex(dates).fillna(0).values

    rec_eq, rec_ret, rec_w, rec_tr, rec_to, rec_cost = [], [], [], [], [], []
    n_trades = n_skipped = 0
    for pos in range(p0, p_end):
        # 1. Performance de la semaine (positions décidées précédemment)
        prev_total = V.sum()
        if pos > p0:
            r = np.nan_to_num(R_arr[pos])
            cash_w = 1 - W.sum(axis=1)
            growth = W @ (1 + r) + cash_w * (1 + rf_arr[pos - 1])
            W = W * (1 + r) / growth[:, None]
            V = V * growth

        # 2. Exécution des rééquilibrages programmés cette semaine
        signal_date = dates[pos - cfg.execution_lag]
        step = pos - p0
        turnover_val, cost_val = 0.0, 0.0
        due = [k for k in range(K) if step == 0 or (step - offsets[k]) % R == 0 and step >= offsets[k]]
        if due:
            for k in due:
                w_cur = pd.Series(W[k], index=tickers)
                target, upper = target_fn(signal_date, w_cur)
                target = target.reindex(tickers).fillna(0).values
                upper = upper.reindex(tickers).fillna(1.0).values
                one_way = 0.5 * np.abs(target - W[k]).sum()
                if step > 0 and one_way < cfg.buffer:
                    new_w = np.minimum(W[k], upper)          # sorties de risque uniquement
                    n_skipped += 1
                else:
                    new_w = target
                    n_trades += 1
                traded = np.abs(new_w - W[k]).sum()
                cost = c * traded * V[k]
                V[k] -= cost
                W[k] = new_w
                turnover_val += 0.5 * traded * V[k]
                cost_val += cost

        total = V.sum()
        rec_eq.append(total)
        rec_ret.append(total / prev_total - 1 if pos > p0 else 0.0)
        rec_w.append((V[:, None] * W).sum(axis=0) / total)
        rec_tr.append(V * K)
        rec_to.append(turnover_val / total)
        rec_cost.append(cost_val / total)
        if progress and (pos - p0) % 20 == 0:
            progress((pos - p0) / max(p_end - p0, 1))

    idx = dates[p0:p_end]
    eq = pd.Series(rec_eq, index=idx, name="equity")
    return BacktestResult(
        equity=eq / eq.iloc[0],
        returns=pd.Series(rec_ret, index=idx, name="return"),
        weights=pd.DataFrame(rec_w, index=idx, columns=tickers),
        tranche_equity=pd.DataFrame(rec_tr, index=idx, columns=[f"Tranche {k + 1}" for k in range(K)]),
        turnover=pd.Series(rec_to, index=idx),
        costs=pd.Series(rec_cost, index=idx),
        n_trades=n_trades, n_skipped=n_skipped,
    )


def static_target(weights: dict[str, float]) -> TargetFn:
    """Portefeuille à poids fixes (benchmarks), rééquilibré selon le même calendrier."""
    def fn(date, w0):
        w = pd.Series(weights, dtype=float)
        return w, pd.Series(1.0, index=w0.index)
    return fn


def model_target(model, cost: float) -> TargetFn:
    """Adaptateur AllocationModel → TargetFn (poids cibles + bornes de tendance)."""
    def fn(date, w0):
        if date not in model.fc.index:
            return pd.Series(0.0, index=w0.index), pd.Series(0.0, index=w0.index)
        return model.weights(date, w0, cost), model.inputs(date).upper
    return fn
