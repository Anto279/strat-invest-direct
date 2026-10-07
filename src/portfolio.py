"""
Allocation moyenne-variance (Markowitz 1952) sous contraintes « Safety-First ».

Problème résolu à chaque rééquilibrage (unités : horizon h = 4 semaines) :

    max_w   w'(μ − r_f 1) − (γ/2) w'Σw − c · Σ_i |w_i − w_i⁰|
    s.c.    0 ≤ w_i ≤ w_max · s_i          (long-only, borne pilotée par la tendance SMA)
            Σ_i w_i ≤ 1                    (le complément est investi en T-Bills)
            √(52/h · w'Σw) ≤ σ_cap          (budget de volatilité ex-ante)

Pourquoi une utilité et non le ratio de Sharpe ? En présence d'un actif sans
risque, le Sharpe est invariant d'échelle (w et λw ont le même Sharpe) : il
fixe la *direction* du portefeuille tangent mais pas le *montant* investi.
L'utilité quadratique détermine les deux ; sans contrainte active, sa solution
est w* = Σ⁻¹(μ − r_f)/γ, colinéaire au portefeuille de Sharpe maximal.

Espérances de rendement — combinaison bayésienne prior/vues :
    π_i = r_f + SR_prior · σ_i            (prior d'équilibre « Sharpe constant »)
    μ_i = (1 − κ) π_i + κ · f_i           (f_i = prévision ARIMA-XGBoost)
Avec κ = 0, l'allocation ne dépend que du risque (portefeuille diversifié
proche de la parité de risque) ; κ mesure la confiance accordée aux prévisions.
Ce rétrécissement (shrinkage) répond au constat de Michaud (1989) : un
optimiseur non contraint est un « maximiseur d'erreurs d'estimation ».

Covariance : Σ = D R D, avec D les volatilités GARCH prévues sur l'horizon et
R la corrélation sur 104 semaines, rétrécie vers l'identité (Ledoit-Wolf 2004).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.config import ModelConfig, PortfolioConfig
from src.trend import trend_score

_EPS = 1e-8


@dataclass
class AllocationInputs:
    mu: pd.Series          # rendement simple espéré sur l'horizon
    prior: pd.Series
    forecast: pd.Series
    sigma: pd.Series       # volatilité horizon
    cov: pd.DataFrame
    corr: pd.DataFrame
    upper: pd.Series       # bornes hautes après filtre de tendance
    trend: pd.Series
    rf_h: float


def solve(mu: np.ndarray, cov: np.ndarray, rf_h: float, upper: np.ndarray, w0: np.ndarray,
          gamma: float, cost: float, vol_cap_h: float | None) -> np.ndarray:
    """Résolution SLSQP avec gradients analytiques ; |x| lissé par √(x² + ε)."""
    n = len(mu)
    ex = mu - rf_h
    active = upper > 1e-6
    if not active.any():
        return np.zeros(n)

    def f(w):
        d = w - w0
        return -(w @ ex - 0.5 * gamma * w @ cov @ w - cost * np.sum(np.sqrt(d * d + _EPS)))

    def g(w):
        d = w - w0
        return -(ex - gamma * cov @ w - cost * d / np.sqrt(d * d + _EPS))

    cons = [{"type": "ineq", "fun": lambda w: 1.0 - w.sum(), "jac": lambda w: -np.ones(n)}]
    if vol_cap_h is not None:
        cons.append({"type": "ineq", "fun": lambda w: vol_cap_h ** 2 - w @ cov @ w,
                     "jac": lambda w: -2 * cov @ w})
    bounds = [(0.0, float(u)) for u in upper]
    x0 = np.clip(w0, 0, upper)
    if x0.sum() > 1:
        x0 = x0 / x0.sum()
    res = minimize(f, x0, jac=g, bounds=bounds, constraints=cons, method="SLSQP",
                   options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(res.x if res.success else x0, 0, upper)
    if w.sum() > 1:
        w = w / w.sum()
    w[w < 1e-4] = 0.0
    return w


class AllocationModel:
    """Assemble μ, Σ et bornes à partir du panel de signaux, puis optimise."""

    def __init__(self, prices: pd.DataFrame, rf: pd.Series, signals: pd.DataFrame,
                 mcfg: ModelConfig, pcfg: PortfolioConfig):
        self.tickers = list(prices.columns)
        self.mcfg, self.pcfg = mcfg, pcfg
        self.h = mcfg.horizon
        self.logret = np.log(prices).diff()
        self.rf = rf
        self.trend = trend_score(prices, mcfg.sma_windows)
        self.fc = signals[pcfg.forecast_source].unstack("ticker").reindex(columns=self.tickers) / 100
        self.sig = signals["sigma_h"].unstack("ticker").reindex(columns=self.tickers) / 100
        self._cache: dict[pd.Timestamp, AllocationInputs] = {}

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fc.dropna(how="all").index

    def inputs(self, date: pd.Timestamp) -> AllocationInputs:
        if date in self._cache:
            return self._cache[date]
        p = self.pcfg
        sigma = self.sig.loc[date].fillna(self.sig.loc[date].median())
        rf_h = float((1 + self.rf.get(date, 0.0)) ** self.h - 1)

        hist = self.logret.loc[:date].tail(self.mcfg.corr_window).dropna()
        z = (hist - hist.mean()) / hist.std()
        lw = LedoitWolf().fit(z.values).covariance_
        d = np.sqrt(np.diag(lw))
        corr = pd.DataFrame(lw / np.outer(d, d), index=self.tickers, columns=self.tickers)

        prior = rf_h + p.prior_sharpe * np.sqrt(self.h / 52) * sigma
        forecast = np.exp(self.fc.loc[date].fillna(0) + 0.5 * sigma ** 2) - 1
        mu = (1 - p.view_weight) * prior + p.view_weight * forecast
        cov = pd.DataFrame(np.outer(sigma, sigma) * corr.values, index=self.tickers, columns=self.tickers)

        trend = self.trend.loc[date].fillna(0.0)
        upper = p.max_weight * (trend if p.use_trend_filter else pd.Series(1.0, index=self.tickers))
        out = AllocationInputs(mu, prior, forecast, sigma, cov, corr, upper, trend, rf_h)
        self._cache[date] = out
        return out

    def weights(self, date: pd.Timestamp, w0: pd.Series | None = None, cost: float = 0.001) -> pd.Series:
        x = self.inputs(date)
        w0v = np.zeros(len(self.tickers)) if w0 is None else w0.reindex(self.tickers).fillna(0).values
        vol_cap_h = self.pcfg.vol_cap * np.sqrt(self.h / 52) if self.pcfg.vol_cap else None
        w = solve(x.mu.values, x.cov.values, x.rf_h, x.upper.values, w0v,
                  self.pcfg.risk_aversion, cost, vol_cap_h)
        return pd.Series(w, index=self.tickers)

    def frontier(self, date: pd.Timestamp, n_points: int = 25, constrained: bool = True) -> pd.DataFrame:
        """Frontière efficiente long-only (annualisée), avec ou sans bornes de tendance."""
        x = self.inputs(date)
        ann = 52 / self.h
        upper = x.upper.values if constrained else np.full(len(self.tickers), 1.0)
        if upper.max() <= 1e-6:
            return pd.DataFrame(columns=["vol", "ret"])
        mu, cov = x.mu.values, x.cov.values
        vols_assets = np.sqrt(np.diag(cov))
        targets = np.linspace(vols_assets.min() * 0.3, vols_assets.max(), n_points)
        out = []
        for tv in targets:
            cons = [{"type": "ineq", "fun": lambda w: 1.0 - w.sum()},
                    {"type": "ineq", "fun": lambda w, tv=tv: tv ** 2 - w @ cov @ w}]
            res = minimize(lambda w: -(w @ mu + (1 - w.sum()) * x.rf_h), np.full(len(mu), 0.0),
                           jac=lambda w: -(mu - x.rf_h), bounds=[(0, u) for u in upper],
                           constraints=cons, method="SLSQP")
            if res.success:
                w = res.x
                out.append({"vol": np.sqrt(w @ cov @ w * ann),
                            "ret": (w @ mu + (1 - w.sum()) * x.rf_h) * ann})
        return pd.DataFrame(out).drop_duplicates().sort_values("vol")
