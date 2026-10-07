"""
Générateur de données SYNTHÉTIQUES pour les tests automatisés uniquement.
Aucune de ces séries n'est utilisée dans l'application ou le README.
Rendements quotidiens corrélés, volatilité GARCH(1,1), facteur de tendance lent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MACRO_TICKERS, RISK_FREE_TICKER, TICKERS


def synthetic_daily(start="2006-01-02", end="2024-12-31", seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n, k = len(idx), len(TICKERS)
    corr = np.full((k, k), 0.2)
    eq = [0, 1, 2, 7]
    for i in eq:
        for j in eq:
            corr[i, j] = 0.75
    corr[3, 4] = corr[4, 3] = 0.85
    for i in eq:
        corr[i, 3] = corr[3, i] = corr[i, 4] = corr[4, i] = -0.2
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)
    vol = np.array([0.18, 0.2, 0.25, 0.07, 0.15, 0.16, 0.2, 0.25]) / np.sqrt(252)
    drift = np.array([0.08, 0.05, 0.05, 0.03, 0.04, 0.05, 0.01, 0.06]) / 252
    trend = np.cumsum(rng.normal(0, 0.002, (n, k)), axis=0)
    trend = 0.0004 * np.tanh(trend / 0.05)
    h = np.ones(k)
    rets = np.empty((n, k))
    for t in range(n):
        z = L @ rng.standard_t(6, k) / np.sqrt(1.5)
        h = 0.02 + 0.08 * (z ** 2) + 0.9 * h
        rets[t] = drift + trend[t] + vol * np.sqrt(h) * z
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=TICKERS)
    vix = 15 * np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.2) + 5
    macro = pd.DataFrame({
        MACRO_TICKERS["VIX"]: vix,
        MACRO_TICKERS["TNX"]: 3 + np.cumsum(rng.normal(0, 0.03, n)) * 0.3,
        MACRO_TICKERS["DXY"]: 90 * np.exp(np.cumsum(rng.normal(0, 0.004, n))),
        RISK_FREE_TICKER: np.clip(2 + np.cumsum(rng.normal(0, 0.02, n)) * 0.3, 0, 6),
    }, index=idx)
    return pd.concat([prices, macro], axis=1)
