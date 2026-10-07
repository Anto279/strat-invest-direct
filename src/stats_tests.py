"""
Outils d'inférence partagés : comparaison de prévisions et régressions
d'efficience. Les variances sont robustes à l'autocorrélation (HAC Newey-West),
indispensable lorsque les cibles se chevauchent (rendements cumulés sur h
semaines évalués chaque semaine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def newey_west_var(x: np.ndarray, lags: int) -> float:
    """Variance de long terme (noyau de Bartlett) de la moyenne d'une série."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    gamma0 = x @ x / n
    v = gamma0
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * (x[k:] @ x[:-k]) / n
    return max(v, 1e-18)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> tuple[float, float]:
    """
    Test de Diebold-Mariano (1995) avec correction petit échantillon de
    Harvey, Leybourne & Newbold (1997). H0 : précision égale.
    Statistique < 0  ⇔  le modèle A a une perte moyenne plus faible.
    p-value bilatérale (loi de Student n−1).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    lags = max(horizon - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    dm = d.mean() / np.sqrt(newey_west_var(d, lags) / n)
    k = horizon
    correction = np.sqrt((n + 1 - 2 * k + k * (k - 1) / n) / n)
    stat = dm * correction
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def oos_r2(y: np.ndarray, pred: np.ndarray, bench: np.ndarray) -> float:
    """R² hors échantillon de Campbell & Thompson (2008) relativement à une prévision de référence."""
    y, pred, bench = map(lambda a: np.asarray(a, float), (y, pred, bench))
    m = np.isfinite(y) & np.isfinite(pred) & np.isfinite(bench)
    sse_m = np.sum((y[m] - pred[m]) ** 2)
    sse_b = np.sum((y[m] - bench[m]) ** 2)
    return float(1 - sse_m / sse_b) if sse_b > 0 else np.nan


def mincer_zarnowitz(realized: np.ndarray, forecast: np.ndarray, lags: int = 4) -> dict:
    """
    Régression d'efficience y = a + b·ŷ + u (erreurs HAC). Une prévision
    non biaisée et efficiente vérifie a = 0, b = 1 (test de Wald joint).
    """
    y, f = np.asarray(realized, float), np.asarray(forecast, float)
    m = np.isfinite(y) & np.isfinite(f)
    X = sm.add_constant(f[m])
    res = sm.OLS(y[m], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    wald = res.wald_test("const = 0, x1 = 1", scalar=True)
    return {"a": float(res.params[0]), "b": float(res.params[1]), "R²": float(res.rsquared),
            "Wald (a=0, b=1) p-value": float(wald.pvalue)}


def information_coefficient(y: np.ndarray, pred: np.ndarray) -> float:
    """Corrélation de rang de Spearman entre prévision et réalisation."""
    s = pd.DataFrame({"y": y, "p": pred}).dropna()
    if len(s) < 10:
        return np.nan
    return float(stats.spearmanr(s.y, s.p).statistic)


def hit_rate(y: np.ndarray, pred: np.ndarray) -> float:
    s = pd.DataFrame({"y": y, "p": pred}).dropna()
    s = s[(s.p != 0)]
    return float((np.sign(s.y) == np.sign(s.p)).mean()) if len(s) else np.nan
