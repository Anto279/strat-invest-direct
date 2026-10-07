"""
Modélisation de la volatilité conditionnelle (famille GARCH) sur les
résidus ARIMA.

Spécification retenue dans la pipeline : GARCH(1,1) à innovations Student-t,
    ε_t = σ_t z_t,  z_t ~ t_ν standardisée
    σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
Alternatives évaluées (onglet recherche) : GARCH-N, GJR-GARCH-t (effet de
levier asymétrique, Glosten-Jagannathan-Runkle 1993), EGARCH-t (Nelson 1991).

Évaluation hors échantillon : les paramètres sont estimés sur l'échantillon
d'apprentissage puis GELÉS ; la variance est filtrée sur l'échantillon de test
et comparée au proxy de variance réalisée ε²_{t}. Fonctions de perte :
QLIKE (robuste au bruit du proxy, Patton 2011) et MSE. Références naïves :
EWMA RiskMetrics (λ = 0.94) et variance historique glissante 26 semaines.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.stats_tests import diebold_mariano, mincer_zarnowitz

SPECS: dict[str, dict] = {
    "GARCH(1,1) Normale": dict(vol="GARCH", p=1, o=0, q=1, dist="normal"),
    "GARCH(1,1) Student-t": dict(vol="GARCH", p=1, o=0, q=1, dist="t"),
    "GJR-GARCH(1,1,1) Student-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
    "EGARCH(1,1) Student-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
}
DEFAULT_SPEC = "GARCH(1,1) Student-t"


def fit_garch(resid, spec: str = DEFAULT_SPEC, last_obs=None, starting_values=None):
    """Estimation par maximum de vraisemblance ; None en cas d'échec."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            am = arch_model(resid, mean="Zero", rescale=False, **SPECS[spec])
            res = am.fit(disp="off", last_obs=last_obs, starting_values=starting_values,
                         options={"maxiter": 200})
            return res if res.convergence_flag == 0 else None
        except Exception:
            return None


def persistence(res) -> float:
    p = res.params
    if "omega" not in p.index:
        return np.nan
    if res.model.volatility.__class__.__name__ == "EGARCH":
        return float(p.get("beta[1]", np.nan))
    return float(p.get("alpha[1]", 0) + p.get("beta[1]", 0) + 0.5 * p.get("gamma[1]", 0))


def horizon_vol(res, horizon: int) -> float:
    """Volatilité du rendement cumulé sur h pas : sqrt(Σ_{k=1..h} σ²_{t+k|t})."""
    f = res.forecast(horizon=horizon, reindex=False)
    return float(np.sqrt(np.sum(f.variance.values[-1])))


def ewma_variance(e: np.ndarray, lam: float = 0.94, init: float | None = None) -> np.ndarray:
    """σ²_{t+1|t} = λ σ²_{t|t-1} + (1-λ) ε²_t ; renvoie la prévision pour chaque t+1."""
    out = np.empty(len(e))
    s2 = np.var(e) if init is None else init
    for t, x in enumerate(e):
        s2 = lam * s2 + (1 - lam) * x * x
        out[t] = s2
    return out


def compare_specs(resid: pd.Series) -> pd.DataFrame:
    """Vraisemblance, critères d'information et persistance de chaque spécification."""
    rows = {}
    for name in SPECS:
        res = fit_garch(resid, name)
        if res is None:
            continue
        pers = persistence(res)
        rows[name] = {
            "Log-vraisemblance": res.loglikelihood, "AIC": res.aic, "BIC": res.bic,
            "Paramètres": len(res.params), "Persistance": pers,
            "Demi-vie (sem.)": np.log(0.5) / np.log(pers) if 0 < pers < 1 else np.inf,
        }
    out = pd.DataFrame(rows).T.astype(float)
    if not out.empty:
        out["Paramètres"] = out["Paramètres"].astype(int)
    return out


def standardized_diagnostics(res, lags: int = 12) -> pd.DataFrame:
    z = pd.Series(res.std_resid).dropna().values
    rows = []
    lb = acorr_ljungbox(z, lags=[lags], return_df=True).iloc[0]
    rows.append({"Test": f"Ljung-Box z (lag {lags})", "H0": "z non autocorrélés", "Statistique": lb.lb_stat, "p-value": lb.lb_pvalue})
    lb2 = acorr_ljungbox(z ** 2, lags=[lags], return_df=True).iloc[0]
    rows.append({"Test": f"Ljung-Box z² (lag {lags})", "H0": "Pas d'ARCH résiduel", "Statistique": lb2.lb_stat, "p-value": lb2.lb_pvalue})
    lm, lm_p, _, _ = het_arch(z, nlags=lags)
    rows.append({"Test": f"ARCH-LM z (lag {lags})", "H0": "Homoscédasticité", "Statistique": lm, "p-value": lm_p})
    jb = stats.jarque_bera(z)
    rows.append({"Test": "Jarque-Bera z", "H0": "Normalité", "Statistique": jb.statistic, "p-value": jb.pvalue})
    out = pd.DataFrame(rows)
    out["Décision (5 %)"] = np.where(out["p-value"] < 0.05, "Rejet H0", "Non-rejet")
    return out


def qlike(realized_var: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """QLIKE (à constante près) : ln σ̂² + ε² / σ̂²."""
    f = np.maximum(forecast_var, 1e-12)
    return np.log(f) + realized_var / f


def out_of_sample(resid: pd.Series, split: int, spec: str = DEFAULT_SPEC) -> dict:
    """
    Paramètres estimés sur [0, split), gelés, puis prévisions à 1 pas sur le test.
    Renvoie séries alignées, tableau de pertes, test DM et régression MZ.
    """
    e = resid.astype(float)
    res = fit_garch(e, spec, last_obs=split)
    if res is None:
        return {}
    fc = res.forecast(start=split - 1, horizon=1, reindex=False).variance.iloc[:, 0]
    garch_var = pd.Series(fc.values[:-1], index=e.index[split:split + len(fc) - 1])
    realized = e.loc[garch_var.index] ** 2

    ew = pd.Series(ewma_variance(e.values, init=np.var(e.values[:split])), index=e.index).shift(1)
    roll = (e ** 2).rolling(26).mean().shift(1)
    cands = {spec: garch_var, "EWMA (λ = 0.94)": ew.loc[garch_var.index], "Historique 26 sem.": roll.loc[garch_var.index]}

    rows = {}
    for name, f in cands.items():
        rows[name] = {"QLIKE": np.nanmean(qlike(realized.values, f.values)),
                      "RMSE (variance)": np.sqrt(np.nanmean((realized.values - f.values) ** 2))}
    table = pd.DataFrame(rows).T
    dm = {name: diebold_mariano(qlike(realized.values, garch_var.values), qlike(realized.values, f.values))
          for name, f in cands.items() if name != spec}
    mz = mincer_zarnowitz(realized.values, garch_var.values)

    test_fit = fit_garch(e.iloc[split:], spec)
    return {"train_result": res, "forecast_var": garch_var, "realized_var": realized,
            "candidates": cands, "losses": table, "dm": dm, "mz": mz, "test_fit": test_fit}
