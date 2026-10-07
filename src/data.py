"""
Chargement et préparation des données de marché.

Choix méthodologiques :
- Prix ajustés (dividendes + splits) : rendements « total return », seule base
  comparable entre ETF actions, obligataires et matières premières.
- Barres hebdomadaires construites à partir des clôtures quotidiennes
  (dernière clôture de la semaine, label vendredi). Contrairement à
  l'intervalle « 1wk » de Yahoo, cela évite les barres partielles de la
  semaine en cours, sources de look-ahead implicite.
- Cache local Parquet : les résultats sont reproductibles hors-ligne et le
  téléchargement n'est effectué qu'à la demande.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_START, MACRO_TICKERS, RISK_FREE_TICKER, TICKERS

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_FILE = CACHE_DIR / "market_weekly.parquet"
WEEK_RULE = "W-FRI"
PERIODS_PER_YEAR = 52


@dataclass
class MarketData:
    prices: pd.DataFrame   # clôtures ajustées hebdomadaires (date x ticker)
    macro: pd.DataFrame    # VIX, TNX (rendement 10 ans, %), DXY
    rf: pd.Series          # taux sans risque hebdomadaire (rendement simple)

    @property
    def returns(self) -> pd.DataFrame:
        """Rendements simples hebdomadaires."""
        return self.prices.pct_change().dropna(how="all")

    @property
    def log_returns(self) -> pd.DataFrame:
        return np.log(self.prices).diff().dropna(how="all")


def _download_daily_close(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise ConnectionError("Téléchargement Yahoo Finance vide.")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        close.columns = tickers
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def to_weekly(daily: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Dernière clôture de chaque semaine ; supprime la semaine en cours si incomplète."""
    daily = daily.dropna(how="all")
    weekly = daily.resample(WEEK_RULE).last()
    today = (today or pd.Timestamp.today()).normalize()
    if len(weekly) and weekly.index[-1] >= today and daily.index[-1] < weekly.index[-1]:
        weekly = weekly.iloc[:-1]
    return weekly


def build_market_data(daily: pd.DataFrame) -> MarketData:
    """Assemble les blocs prix / macro / taux sans risque à partir des clôtures quotidiennes."""
    weekly = to_weekly(daily)
    prices = weekly[TICKERS].ffill(limit=2).dropna(how="any")

    macro_cols = {v: k for k, v in MACRO_TICKERS.items()}
    macro = weekly[list(macro_cols)].rename(columns=macro_cols).ffill().reindex(prices.index)

    irx = weekly[RISK_FREE_TICKER].ffill().reindex(prices.index).ffill().fillna(0.0)
    rf = (1.0 + irx.clip(lower=0.0) / 100.0) ** (1.0 / PERIODS_PER_YEAR) - 1.0
    rf.name = "rf"
    return MarketData(prices=prices, macro=macro, rf=rf)


def save_cache(md: MarketData, path: Path = CACHE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        {"price": md.prices, "macro": md.macro, "rf": md.rf.to_frame()}, axis=1
    )
    frame.columns = [f"{a}|{b}" for a, b in frame.columns]
    frame.to_parquet(path)


def load_cache(path: Path = CACHE_FILE) -> MarketData | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    blocks: dict[str, dict[str, pd.Series]] = {}
    for col in frame.columns:
        block, name = col.split("|", 1)
        blocks.setdefault(block, {})[name] = frame[col]
    return MarketData(
        prices=pd.DataFrame(blocks["price"]),
        macro=pd.DataFrame(blocks["macro"]),
        rf=pd.Series(blocks["rf"]["rf"], name="rf"),
    )


def load_market_data(refresh: bool = False, start: str = DATA_START) -> MarketData:
    """Retourne les données depuis le cache, ou les télécharge si demandé / absent."""
    cached = None if refresh else load_cache()
    if cached is not None:
        return cached
    try:
        daily = _download_daily_close(TICKERS + list(MACRO_TICKERS.values()) + [RISK_FREE_TICKER], start)
        md = build_market_data(daily)
    except Exception:
        fallback = load_cache()
        if fallback is not None:
            return fallback
        raise
    save_cache(md)
    return md
