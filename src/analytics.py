"""
Statistiques descriptives de l'univers et analyse de la diversification.

- Statistiques marginales : moments, normalité (Jarque-Bera), drawdown.
- Clustering hiérarchique sur la distance de corrélation
      d_ij = √(½ (1 − ρ_ij))      (métrique propre, Mantegna 1999)
  liaison de Ward ; le nombre de groupes est choisi par le score de
  silhouette moyen.
- Mesures de diversification : corrélation moyenne, nombre effectif de
  paris (exponentielle de l'entropie du spectre de la matrice de corrélation,
  cf. Meucci 2009 / Roy & Vetterli 2007) et ratio de diversification du
  portefeuille équipondéré (Choueifaty & Coignard 2008).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from src.metrics import drawdown

P = 52


def summary_table(prices: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    rets = prices.pct_change().dropna()
    lr = np.log(prices).diff().dropna()
    rf_al = rf.shift(1).reindex(rets.index).fillna(0)
    rows = {}
    for t in prices.columns:
        r = rets[t]
        years = len(r) / P
        cagr = (prices[t].iloc[-1] / prices[t].iloc[0]) ** (1 / years) - 1
        vol = r.std() * np.sqrt(P)
        rows[t] = {
            "Rendement annualisé": cagr,
            "Volatilité annualisée": vol,
            "Sharpe": (r - rf_al).mean() * P / vol,
            "Max Drawdown": drawdown(prices[t]).min(),
            "Asymétrie": stats.skew(lr[t]),
            "Kurtosis (excès)": stats.kurtosis(lr[t]),
            "Jarque-Bera p": stats.jarque_bera(lr[t]).pvalue,
            "AC(1) rendements": lr[t].autocorr(1),
        }
    return pd.DataFrame(rows).T


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    d = np.sqrt(np.clip(0.5 * (1 - corr.values), 0, None))
    np.fill_diagonal(d, 0)
    return d


def hierarchical_clusters(corr: pd.DataFrame, k_range=range(2, 6)) -> dict:
    d = correlation_distance(corr)
    Z = linkage(squareform(d, checks=False), method="ward")
    scores = {}
    for k in k_range:
        if k >= len(corr):
            break
        labels = fcluster(Z, k, criterion="maxclust")
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(d, labels, metric="precomputed")
    k_best = max(scores, key=scores.get)
    labels = pd.Series(fcluster(Z, k_best, criterion="maxclust"), index=corr.index, name="Cluster")
    order = [corr.index[i] for i in leaves_list(Z)]
    return {"linkage": Z, "labels": labels, "order": order, "silhouette": pd.Series(scores), "k": k_best}


def effective_number_of_bets(corr: pd.DataFrame) -> float:
    ev = np.clip(np.linalg.eigvalsh(corr.values), 1e-12, None)
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def diversification_ratio(cov: pd.DataFrame, w: np.ndarray) -> float:
    vols = np.sqrt(np.diag(cov.values))
    return float(w @ vols / np.sqrt(w @ cov.values @ w))


def diversification_summary(log_returns: pd.DataFrame) -> dict:
    corr = log_returns.corr()
    n = len(corr)
    off = corr.values[~np.eye(n, dtype=bool)]
    ev = np.sort(np.linalg.eigvalsh(corr.values))[::-1]
    return {
        "Corrélation moyenne": float(off.mean()),
        "Nombre effectif de paris": effective_number_of_bets(corr),
        "Ratio de diversification (1/N)": diversification_ratio(log_returns.cov(), np.full(n, 1 / n)),
        "Variance expliquée par la 1re composante": float(ev[0] / ev.sum()),
        "Nombre d'actifs": n,
    }


def rolling_correlation(log_returns: pd.DataFrame, a: str, b: str, window: int = 52) -> pd.Series:
    return log_returns[a].rolling(window).corr(log_returns[b])
