"""
Tests des invariants méthodologiques (exécution : pytest -q).
Données synthétiques uniquement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest, static_target
from src.config import BacktestConfig, TICKERS
from src.data import build_market_data, to_weekly
from src.features import asset_features
from src.models.hybrid import ResidualBooster, residual_target
from src.portfolio import solve
from src.trend import trend_score
from tests.synthetic import synthetic_daily


@pytest.fixture(scope="module")
def md():
    return build_market_data(synthetic_daily("2006-01-02", "2014-12-31"))


def test_static_backtest_matches_buy_and_hold(md):
    """100 % sur un actif, sans coûts ni décalage : la valeur suit exactement le prix."""
    cfg = BacktestConfig(cost_bps=0, buffer=0, execution_lag=0, n_tranches=1)
    res = run_backtest(md.returns, md.rf, static_target({"SPY": 1.0}), cfg, start=md.returns.index[0])
    px = md.prices["SPY"].loc[res.equity.index]
    np.testing.assert_allclose(res.equity.values, (px / px.iloc[0]).values, rtol=1e-10)


def test_cash_earns_risk_free(md):
    cfg = BacktestConfig(cost_bps=0, buffer=0, execution_lag=0, n_tranches=1)
    res = run_backtest(md.returns, md.rf, static_target({}), cfg, start=md.returns.index[0])
    expected = (1 + md.rf.shift(1).reindex(res.equity.index).fillna(0).iloc[1:]).prod()
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-10)


def test_costs_reduce_value(md):
    base = BacktestConfig(cost_bps=0, buffer=0, execution_lag=0)
    costly = BacktestConfig(cost_bps=50, buffer=0, execution_lag=0)
    w = {t: 1 / len(TICKERS) for t in TICKERS}
    a = run_backtest(md.returns, md.rf, static_target(w), base, start=md.returns.index[0])
    b = run_backtest(md.returns, md.rf, static_target(w), costly, start=md.returns.index[0])
    assert b.equity.iloc[-1] < a.equity.iloc[-1]
    assert b.costs.sum() > 0


def test_features_are_causal(md):
    """Modifier le futur ne change pas les features passées."""
    p = md.prices["SPY"]
    cut = p.index[300]
    f1 = asset_features(p, md.macro).loc[:cut]
    p2 = p.copy()
    p2.loc[p2.index > cut] *= 1.5
    m2 = md.macro.copy()
    m2.loc[m2.index > cut] *= 2
    f2 = asset_features(p2, m2).loc[:cut]
    pd.testing.assert_frame_equal(f1, f2)


def test_residual_target_alignment():
    e = pd.Series(np.arange(10, dtype=float))
    t = residual_target(e, 3)
    assert t.iloc[0] == 1 + 2 + 3
    assert t.iloc[-3:].isna().all()


def test_trend_score_bounds(md):
    s = trend_score(md.prices, (30, 40, 52)).dropna()
    assert ((s >= 0) & (s <= 1)).all().all()
    assert set(np.unique(s.values.round(6))) <= {0.0, round(1 / 3, 6), round(2 / 3, 6), 1.0}


def test_optimizer_constraints():
    rng = np.random.default_rng(0)
    n = 8
    A = rng.normal(size=(n, n))
    cov = (A @ A.T) / n * 1e-3 + np.eye(n) * 1e-4
    mu = rng.normal(0.01, 0.01, n)
    upper = np.array([0.3, 0.3, 0, 0.1, 0.3, 0.3, 0.3, 0.2])
    vol_cap = 0.02
    w = solve(mu, cov, 0.001, upper, np.zeros(n), gamma=2.0, cost=0.001, vol_cap_h=vol_cap)
    assert (w >= -1e-9).all() and (w <= upper + 1e-9).all()
    assert w.sum() <= 1 + 1e-9
    assert np.sqrt(w @ cov @ w) <= vol_cap * 1.001
    assert w[2] == 0


def test_all_bearish_goes_to_cash():
    w = solve(np.full(3, 0.05), np.eye(3) * 1e-3, 0.0, np.zeros(3), np.zeros(3), 5.0, 0.001, None)
    assert np.allclose(w, 0)


def test_buffer_never_blocks_risk_exit(md):
    """Turnover proposé sous le buffer, mais borne de tendance abaissée : la position est coupée."""
    tickers = list(md.returns.columns)
    state = {"n": 0}

    def fn(date, w0):
        state["n"] += 1
        target = pd.Series(0.0, index=tickers)
        target["SPY"] = 0.10
        upper = pd.Series(1.0, index=tickers)
        if state["n"] > 1:
            target["SPY"] = 0.0          # turnover 5 % < buffer 20 %
            upper["SPY"] = 0.0           # mais sortie de risque
        return target, upper

    cfg = BacktestConfig(cost_bps=0, buffer=0.20, execution_lag=0, n_tranches=1)
    res = run_backtest(md.returns.iloc[:20], md.rf, fn, cfg, start=md.returns.index[0])
    assert res.weights["SPY"].iloc[-1] == 0


def test_booster_calibration_shrinks_noise():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(250, 6)), columns=list("abcdef"))
    y = pd.Series(rng.normal(size=250))
    b = ResidualBooster(n_seeds=1).fit(X, y, horizon=4)
    assert 0 <= b.shrink < 0.5


def test_weekly_drops_incomplete_week():
    idx = pd.bdate_range("2024-01-01", "2024-01-17")  # mercredi : semaine en cours
    daily = pd.DataFrame({"A": np.arange(len(idx), dtype=float)}, index=idx)
    w = to_weekly(daily, today=pd.Timestamp("2024-01-17"))
    assert w.index[-1] == pd.Timestamp("2024-01-12")
