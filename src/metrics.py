"""
Mesures de performance (fréquence hebdomadaire, 52 périodes par an).

Conventions :
- Rendement annualisé : taux de croissance géométrique (CAGR).
- Volatilité annualisée : écart-type des rendements hebdomadaires × √52.
- Sharpe : moyenne arithmétique des rendements EXCÉDENTAIRES (vs T-Bills)
  × 52 / volatilité annualisée. (L'ancienne formule (CAGR − 2 %)/σ mélangeait
  moyenne géométrique et taux sans risque constant.)
- Calmar : CAGR / |drawdown maximal|.
- Erreur-type du Sharpe : Lo (2002), hypothèse i.i.d. ; t-stat = SR / SE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P = 52


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def performance(returns: pd.Series, rf: pd.Series) -> dict:
    r = returns.iloc[1:] if returns.iloc[0] == 0 else returns
    rf_al = rf.shift(1).reindex(r.index).fillna(0)
    eq = (1 + r).cumprod()
    years = len(r) / P
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(P)
    ex = r - rf_al
    sharpe = ex.mean() * P / vol if vol > 0 else np.nan
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(P)
    mdd = drawdown(pd.concat([pd.Series([1.0]), eq])).min()
    sr_w = ex.mean() / r.std()
    se = np.sqrt((1 + 0.5 * sr_w ** 2) / len(r)) * np.sqrt(P)
    return {
        "Rendement annualisé": cagr,
        "Volatilité annualisée": vol,
        "Ratio de Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Ratio de Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Ratio de Sortino": ex.mean() * P / downside if downside > 0 else np.nan,
        "t-stat Sharpe (Lo 2002)": sharpe / se if se > 0 else np.nan,
        "Semaines positives": (r > 0).mean(),
    }


def relative(returns: pd.Series, bench: pd.Series) -> dict:
    df = pd.concat([returns, bench], axis=1).dropna().iloc[1:]
    df.columns = ["s", "b"]
    beta = df.cov().iloc[0, 1] / df.b.var()
    active = df.s - df.b
    te = active.std() * np.sqrt(P)
    return {"Bêta": beta, "Tracking error": te,
            "Ratio d'information": active.mean() * P / te if te > 0 else np.nan,
            "Corrélation": df.s.corr(df.b)}


def calendar_returns(equity: pd.Series) -> pd.Series:
    y = equity.resample("YE").last()
    first = equity.iloc[0]
    out = y.pct_change()
    out.iloc[0] = y.iloc[0] / first - 1
    out.index = out.index.year
    return out


def rolling_sharpe(returns: pd.Series, rf: pd.Series, window: int = 52) -> pd.Series:
    ex = returns - rf.shift(1).reindex(returns.index).fillna(0)
    return ex.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(P)
