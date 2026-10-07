"""
Filtre de tendance par moyennes mobiles simples de long terme.

Score de tendance multi-horizon :
    s_{i,t} = (1/|L|) · Σ_{L} 1{ P_{i,t} > SMA_L(P_i)_t },   L ∈ {30, 40, 52} semaines

- La SMA 40 semaines correspond à la moyenne 10 mois / 200 jours de
  Faber (2007, « A Quantitative Approach to Tactical Asset Allocation »).
- Moyenner plusieurs horizons rend le signal moins dépendant d'un paramètre
  unique (risque de sur-optimisation) et lisse les allers-retours autour
  d'une seule moyenne (whipsaws).
- Le score borne l'allocation : w_i ≤ w_max · s_i. Une tendance baissière
  sur tous les horizons interdit toute détention de l'actif (passage en cash).
  Le filtre ne force jamais l'achat : il agit comme une contrainte, pas
  comme une prévision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sma(prices: pd.DataFrame | pd.Series, window: int):
    return prices.rolling(window, min_periods=window).mean()


def trend_score(prices: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """Score ∈ {0, 1/|L|, …, 1} ; NaN tant que la plus longue SMA n'est pas définie."""
    signals = [(prices > sma(prices, w)).astype(float).where(sma(prices, w).notna()) for w in windows]
    return sum(signals) / len(windows)


def conditional_stats(price: pd.Series, score: pd.Series, horizon: int = 4) -> pd.DataFrame:
    """
    Rendements futurs (horizon semaines, non chevauchants) conditionnés au signal :
    tendance haussière (score = 1) vs baissière (score = 0). Test de Welch
    (variances inégales) sur la différence de moyennes.
    """
    fwd = np.log(price.shift(-horizon) / price)
    df = pd.DataFrame({"fwd": fwd, "s": score}).dropna().iloc[::horizon]
    up, down = df.loc[df.s == 1, "fwd"], df.loc[df.s == 0, "fwd"]
    ann = 52 / horizon
    rows = {}
    for name, x in (("Tendance haussière (s = 1)", up), ("Tendance baissière (s = 0)", down)):
        rows[name] = {
            "Observations": len(x),
            "Rendement annualisé": x.mean() * ann if len(x) else np.nan,
            "Volatilité annualisée": x.std() * np.sqrt(ann) if len(x) > 1 else np.nan,
            "Sharpe (brut)": (x.mean() / x.std() * np.sqrt(ann)) if len(x) > 1 and x.std() > 0 else np.nan,
        }
    out = pd.DataFrame(rows).T
    if len(up) > 2 and len(down) > 2:
        t, p = stats.ttest_ind(up, down, equal_var=False)
        out.attrs["welch_t"], out.attrs["welch_p"] = float(t), float(p)
        lv = stats.levene(up, down)
        out.attrs["levene_p"] = float(lv.pvalue)
    return out
