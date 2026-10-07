"""
Ingénierie des features pour le modèle XGBoost sur résidus ARIMA.

Principes :
1. Causalité stricte : chaque feature à la date t n'utilise que l'information
   disponible à la clôture de t (fenêtres glissantes « backward-looking »).
2. Stationnarité : uniquement des rendements, ratios et écarts relatifs.
   Les niveaux bruts (prix, MACD en dollars, niveau du VIX) sont proscrits :
   leur distribution dérive dans le temps et un arbre entraîné sur une
   fenêtre passée extrapolerait hors de son support.
3. Pas de normalisation : les arbres de décision sont invariants à toute
   transformation monotone des features. Un z-score glissant n'apporte rien
   et peut introduire de la fuite d'information s'il est mal fenêtré.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Documentation affichée dans l'interface : définition + rationnel économique.
FEATURE_DOC: dict[str, tuple[str, str, str]] = {
    "mom_1w": ("Momentum", "ln(P_t / P_{t-1})", "Réversion de court terme (microstructure, liquidité)."),
    "mom_4w": ("Momentum", "ln(P_t / P_{t-4})", "Réversion mensuelle documentée (Jegadeesh 1990)."),
    "mom_12w": ("Momentum", "ln(P_t / P_{t-12})", "Momentum intermédiaire (time-series momentum)."),
    "mom_26w": ("Momentum", "ln(P_t / P_{t-26})", "Momentum 6 mois (Moskowitz, Ooi & Pedersen 2012)."),
    "dist_sma40": ("Tendance", "P_t / SMA40_t − 1", "Extension par rapport à la tendance de long terme."),
    "drawdown_52w": ("Tendance", "P_t / max_{52s}(P) − 1", "Distance au plus haut annuel (ancrage, capitulation)."),
    "vol_12w": ("Volatilité", "σ(r, 12s) · √52", "Niveau de risque réalisé récent."),
    "vol_ratio": ("Volatilité", "σ(r, 4s) / σ(r, 26s)", "Choc de volatilité (régime de stress naissant)."),
    "rsi_14": ("Oscillateur", "RSI Wilder 14s, centré", "Sur-achat / sur-vente."),
    "bb_pctb": ("Oscillateur", "%B Bollinger (20s, 2σ)", "Position du prix dans son enveloppe de volatilité."),
    "macd_hist": ("Oscillateur", "(MACD − signal) / P_t", "Accélération du momentum, normalisée par le prix."),
    "vix_chg_4w": ("Macro", "ln(VIX_t / VIX_{t-4})", "Variation de l'aversion au risque implicite."),
    "vix_z52": ("Macro", "z-score 52s du VIX", "Stress relatif à l'année écoulée."),
    "tnx_chg_4w": ("Macro", "Δ 4s du taux 10 ans (pts)", "Choc de taux (duration, REITs, or)."),
    "dxy_chg_4w": ("Macro", "ln(DXY_t / DXY_{t-4})", "Dollar : matières premières, émergents."),
}
FEATURES: list[str] = list(FEATURE_DOC)


def _rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def asset_features(price: pd.Series, macro: pd.DataFrame) -> pd.DataFrame:
    """Matrice de features causales pour un actif (index = dates hebdomadaires)."""
    lp = np.log(price)
    r = lp.diff()
    f = pd.DataFrame(index=price.index)
    for k in (1, 4, 12, 26):
        f[f"mom_{k}w"] = lp - lp.shift(k)
    f["dist_sma40"] = price / price.rolling(40).mean() - 1
    f["drawdown_52w"] = price / price.rolling(52).max() - 1
    f["vol_12w"] = r.rolling(12).std() * np.sqrt(52)
    f["vol_ratio"] = r.rolling(4).std() / r.rolling(26).std()
    f["rsi_14"] = _rsi(price) / 100 - 0.5
    mid, sd = price.rolling(20).mean(), price.rolling(20).std()
    f["bb_pctb"] = (price - (mid - 2 * sd)) / (4 * sd)
    macd = price.ewm(span=12, adjust=False).mean() - price.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / price

    m = macro.reindex(price.index).ffill()
    f["vix_chg_4w"] = np.log(m["VIX"] / m["VIX"].shift(4))
    f["vix_z52"] = (m["VIX"] - m["VIX"].rolling(52).mean()) / m["VIX"].rolling(52).std()
    f["tnx_chg_4w"] = m["TNX"] - m["TNX"].shift(4)
    f["dxy_chg_4w"] = np.log(m["DXY"] / m["DXY"].shift(4))
    return f[FEATURES].replace([np.inf, -np.inf], np.nan)
