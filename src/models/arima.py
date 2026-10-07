"""
Modélisation ARIMA des rendements hebdomadaires.

Démarche de Box-Jenkins rendue entièrement explicite :
1. Ordre d'intégration d : tests ADF (H0 : racine unitaire) et KPSS
   (H0 : stationnarité) sur le log-prix puis sur les log-rendements. Les deux
   tests ont des hypothèses nulles opposées ; la conclusion n'est retenue que
   lorsqu'ils concordent. Un log-prix I(1) implique de modéliser les
   rendements avec d = 0 (ARMA(p, q) sur r_t ⇔ ARIMA(p, 1, q) sur ln P_t).
2. Ordres (p, q) : recherche exhaustive sur une grille {0..3}², sélection par
   BIC (Schwarz 1978). Le BIC est convergent : il retrouve le vrai ordre quand
   n → ∞, alors que l'AIC/AICc (Hurvich & Tsai 1989, reporté pour contrôle)
   sur-paramètre systématiquement. Sur des rendements au signal très faible,
   l'AICc sélectionne fréquemment des ARMA(p, p) dont les racines AR et MA se
   compensent presque : modèle instable, prévisions bruitées.
3. Diagnostics des résidus : Ljung-Box (bruit blanc), Jarque-Bera (normalité),
   ARCH-LM d'Engle (hétéroscédasticité conditionnelle → motive le GARCH).

Les rendements sont exprimés en % (×100) pour le conditionnement numérique.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats


def _quiet():
    warnings.simplefilter("ignore")


def stationarity_tests(series: pd.Series, alpha: float = 0.05) -> dict:
    """ADF + KPSS (constante) ; renvoie statistiques, p-values et verdict."""
    x = pd.Series(series).dropna().values
    with warnings.catch_warnings():
        _quiet()
        adf_stat, adf_p, *_ = adfuller(x, regression="c", autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
    if adf_p < alpha and kpss_p > alpha:
        verdict = "Stationnaire"
    elif adf_p >= alpha and kpss_p <= alpha:
        verdict = "Non stationnaire (racine unitaire)"
    else:
        verdict = "Ambigu"
    return {"ADF stat": adf_stat, "ADF p-value": adf_p,
            "KPSS stat": kpss_stat, "KPSS p-value": kpss_p, "Verdict": verdict}


def integration_order(log_price: pd.Series, alpha: float = 0.05) -> tuple[int, pd.DataFrame]:
    """Détermine d par différenciations successives (max 2)."""
    rows, d, x = {}, 0, pd.Series(log_price).dropna()
    for d in range(3):
        res = stationarity_tests(x, alpha)
        rows[f"d = {d}" + (" (log-prix)" if d == 0 else " (log-rendements)" if d == 1 else "")] = res
        if res["Verdict"] == "Stationnaire":
            break
        x = x.diff().dropna()
    out = pd.DataFrame(rows).T
    num = ["ADF stat", "ADF p-value", "KPSS stat", "KPSS p-value"]
    out[num] = out[num].astype(float)
    return d, out


def fit_arma(y: np.ndarray, order: tuple[int, int], start_params=None):
    """ARMA(p, q) avec constante sur les rendements (d = 0)."""
    with warnings.catch_warnings():
        _quiet()
        model = ARIMA(y, order=(order[0], 0, order[1]), trend="c")
        try:
            return model.fit(start_params=start_params) if start_params is not None else model.fit()
        except Exception:
            return model.fit()


def _aicc(res, n: int) -> float:
    k = len(res.params)
    return res.aic + (2 * k * (k + 1)) / max(n - k - 1, 1)


def order_grid(y: np.ndarray, max_p: int = 3, max_q: int = 3) -> pd.DataFrame:
    """Critères d'information pour chaque (p, q) de la grille."""
    n, rows = len(y), []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                res = fit_arma(y, (p, q))
                rows.append({"p": p, "q": q, "AIC": res.aic, "AICc": _aicc(res, n),
                             "BIC": res.bic, "LogLik": res.llf})
            except Exception:
                rows.append({"p": p, "q": q, "AIC": np.nan, "AICc": np.nan, "BIC": np.nan, "LogLik": np.nan})
    return pd.DataFrame(rows)


def select_order(y: np.ndarray, max_p: int = 3, max_q: int = 3, criterion: str = "BIC") -> tuple[int, int]:
    grid = order_grid(y, max_p, max_q).dropna(subset=[criterion])
    if grid.empty:
        return (0, 0)
    best = grid.loc[grid[criterion].idxmin()]
    return int(best.p), int(best.q)


def forecast_sum(res, horizon: int) -> float:
    """Somme des prévisions 1..h pas : prévision du log-rendement cumulé sur l'horizon."""
    return float(np.sum(res.forecast(steps=horizon)))


def residual_diagnostics(resid: np.ndarray, lags=(4, 8, 12)) -> pd.DataFrame:
    """Batterie de tests sur les résidus ; H0 indiquée pour chaque ligne."""
    e = pd.Series(resid).dropna().values
    rows = []
    lb = acorr_ljungbox(e, lags=list(lags), return_df=True)
    for lag, r in lb.iterrows():
        rows.append({"Test": f"Ljung-Box (lag {lag})", "H0": "Résidus non autocorrélés",
                     "Statistique": r.lb_stat, "p-value": r.lb_pvalue})
    lb2 = acorr_ljungbox(e ** 2, lags=[max(lags)], return_df=True).iloc[0]
    rows.append({"Test": f"Ljung-Box résidus² (lag {max(lags)})", "H0": "Pas d'effet ARCH",
                 "Statistique": lb2.lb_stat, "p-value": lb2.lb_pvalue})
    lm, lm_p, _, _ = het_arch(e, nlags=max(lags))
    rows.append({"Test": f"ARCH-LM d'Engle (lag {max(lags)})", "H0": "Homoscédasticité",
                 "Statistique": lm, "p-value": lm_p})
    jb = stats.jarque_bera(e)
    rows.append({"Test": "Jarque-Bera", "H0": "Normalité", "Statistique": jb.statistic, "p-value": jb.pvalue})
    out = pd.DataFrame(rows)
    out["Décision (5 %)"] = np.where(out["p-value"] < 0.05, "Rejet H0", "Non-rejet")
    return out


def coefficient_table(res) -> pd.DataFrame:
    names = res.model.param_names
    return pd.DataFrame({"Coefficient": res.params, "Erreur std.": res.bse,
                         "z": res.tvalues, "p-value": res.pvalues}, index=names)
