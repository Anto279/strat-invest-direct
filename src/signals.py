"""
Génération walk-forward des signaux : à chaque semaine t et pour chaque actif,
ré-estimation sur la fenêtre glissante [t − W + 1, t] uniquement, puis
prévision de l'horizon (t, t + h].

Sorties (en % de log-rendement sur l'horizon) :
    arima_mu   prévision ARIMA cumulée
    xgb_mu     correction XGBoost des résidus (après calibration β)
    xgb_shrink facteur de calibration β ∈ [0, 1] estimé hors échantillon
    hybrid_mu  arima_mu + xgb_mu
    hist_mu    moyenne historique de la fenêtre × h (référence naïve)
    sigma_h    volatilité GARCH du rendement cumulé sur h semaines
    realized   log-rendement réalisé sur (t, t + h]  — évaluation uniquement

Le panel est calculé une fois, mis en cache sur disque, puis partagé par
toutes les tranches et toutes les variantes du backtest (le signal à la date t
ne dépend pas du portefeuille).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import ModelConfig
from src.data import CACHE_DIR, MarketData
from src.features import asset_features
from src.models.arima import fit_arma, forecast_sum, select_order
from src.models.garch import ewma_variance, fit_garch, horizon_vol, persistence
from src.models.hybrid import ResidualBooster, residual_target


def walk_forward_asset(ticker: str, price: pd.Series, macro: pd.DataFrame, cfg: ModelConfig,
                       start_date: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Un processus = un actif : on interdit le multi-threading BLAS/OpenMP interne
    # pour éviter la sur-souscription des cœurs lors de la parallélisation joblib.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _walk_forward_asset(ticker, price, macro, cfg, start_date)


def _walk_forward_asset(ticker, price, macro, cfg, start_date):
    h, W = cfg.horizon, cfg.train_window
    lr = 100 * np.log(price).diff()
    X = asset_features(price, macro)
    realized = lr.rolling(h).sum().shift(-h)
    n = len(price)
    start = W + 1
    if start_date is not None:
        start = max(start, int(price.index.searchsorted(start_date)))

    rows, importances = [], {}
    order, last_sel, arma_params, garch_params = (0, 0), -10 ** 9, None, None
    booster, last_fit = ResidualBooster(cfg.xgb_n_estimators, cfg.xgb_max_depth, cfg.xgb_learning_rate,
                                        cfg.xgb_n_seeds), -10 ** 9

    for t in range(start, n):
        y = lr.iloc[t - W + 1: t + 1]
        if t - last_sel >= cfg.arima_reselect_every:
            new_order = select_order(y.values, cfg.arima_max_p, cfg.arima_max_q, cfg.arima_criterion)
            if new_order != order:
                arma_params = None
            order, last_sel = new_order, t
        res = fit_arma(y.values, order, arma_params)
        arma_params = res.params
        arima_mu = forecast_sum(res, h)

        burn = max(order) + 1
        e = pd.Series(res.resid, index=y.index).iloc[burn:]

        g = fit_garch(e.values, starting_values=garch_params)
        if g is not None and persistence(g) < 0.999:
            garch_params = g.params.values
            sigma_h, pers = horizon_vol(g, h), persistence(g)
        else:
            garch_params = None
            sigma_h, pers = float(np.sqrt(h * ewma_variance(e.values)[-1])), np.nan

        if t - last_fit >= cfg.xgb_refit_every:
            tgt = residual_target(e, h).dropna()
            Xtr = X.loc[tgt.index]
            if len(tgt) >= 52:
                booster.fit(Xtr, tgt, horizon=h)
                importances[price.index[t]] = booster.shap_importance(Xtr)
                last_fit = t
        xgb_mu = float(booster.predict(X.iloc[[t]])[0]) if booster.models else 0.0

        rows.append({
            "date": price.index[t], "ticker": ticker,
            "arima_mu": arima_mu, "xgb_mu": xgb_mu, "hybrid_mu": arima_mu + xgb_mu,
            "hist_mu": float(y.mean() * h), "sigma_h": sigma_h, "garch_persistence": pers,
            "xgb_shrink": booster.shrink if booster.models else np.nan,
            "p": order[0], "q": order[1], "realized": realized.iloc[t],
        })

    imp = pd.DataFrame(importances).T
    imp.index.name = "date"
    imp["ticker"] = ticker
    return pd.DataFrame(rows), imp


def _data_fingerprint(md: MarketData) -> str:
    h = hashlib.sha1()
    h.update(pd.util.hash_pandas_object(md.prices.round(6), index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(md.macro.round(6).fillna(0), index=True).values.tobytes())
    return h.hexdigest()[:12]


def data_fingerprint(md: MarketData) -> str:
    return _data_fingerprint(md)


def cache_paths(md: MarketData, cfg: ModelConfig, start_date=None) -> tuple[Path, Path]:
    key = f"{cfg.key()}_{_data_fingerprint(md)}_{pd.Timestamp(start_date).date() if start_date else 'all'}"
    return CACHE_DIR / f"signals_{key}.parquet", CACHE_DIR / f"importance_{key}.parquet"


def compute_signal_panel(md: MarketData, cfg: ModelConfig, n_jobs: int = -1, progress=None,
                         use_cache: bool = True, start_date=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel (date, ticker) des prévisions + panel des importances SHAP ; mis en cache."""
    f_sig, f_imp = cache_paths(md, cfg, start_date)
    if use_cache and f_sig.exists() and f_imp.exists():
        return pd.read_parquet(f_sig), pd.read_parquet(f_imp)

    tickers = list(md.prices.columns)
    jobs = (delayed(walk_forward_asset)(t, md.prices[t], md.macro, cfg,
                                        pd.Timestamp(start_date) if start_date else None) for t in tickers)
    sigs, imps = [], []
    for i, (s, im) in enumerate(Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs)):
        sigs.append(s)
        imps.append(im)
        if progress:
            progress((i + 1) / len(tickers))
    signals = pd.concat(sigs).set_index(["date", "ticker"]).sort_index()
    importance = pd.concat(imps).reset_index().set_index(["date", "ticker"]).sort_index()
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    signals.to_parquet(f_sig)
    importance.to_parquet(f_imp)
    return signals, importance
