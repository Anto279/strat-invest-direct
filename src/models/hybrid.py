"""
Correction non linéaire des résidus ARIMA par gradient boosting (XGBoost).

Décomposition hybride (Zhang 2003) :
    r_{t→t+h} = L̂_{t,h} + N_{t,h} + η
où L̂ est la prévision linéaire ARIMA cumulée sur h pas et N la composante
non linéaire, apprise sur les résidus d'estimation ARIMA :
    cible_j = Σ_{k=1..h} e_{j+k},   features X_j (connues en j).

Garde-fous contre le sur-apprentissage (signal/bruit très faible) :
- arbres peu profonds (profondeur 2 : interactions d'ordre 2 au plus),
- learning rate faible, sous-échantillonnage lignes/colonnes,
- min_child_weight élevé et régularisation L2,
- moyenne d'un ensemble de graines (réduction de variance),
- seuls les échantillons dont la cible est entièrement observée à la date
  de décision sont utilisés (j + h ≤ t) : aucune fuite d'information ;
- CALIBRATION hors échantillon : à chaque ré-entraînement, une validation
  croisée par blocs contigus purgés (embargo de h semaines de part et d'autre
  du bloc de test, cf. López de Prado 2018) produit des prédictions
  out-of-fold p̃. La pente de la régression y = β p̃ (bornée à [0, 1]) est le
  facteur de rétrécissement optimal au sens des moindres carrés : la
  prédiction finale vaut β · f(X). Sans pouvoir prédictif hors échantillon,
  β → 0 et la correction s'éteint d'elle-même au lieu d'injecter du bruit
  dans l'optimiseur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


class ResidualBooster:
    def __init__(self, n_estimators: int = 100, max_depth: int = 2, learning_rate: float = 0.01,
                 n_seeds: int = 3):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=40, reg_lambda=20.0,
                           tree_method="hist", n_jobs=1, objective="reg:squarederror")
        self.n_seeds = n_seeds
        self.models: list[xgb.XGBRegressor] = []
        self.features: list[str] = []
        self.shrink: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series, horizon: int | None = None, n_folds: int = 4) -> "ResidualBooster":
        self.features = list(X.columns)
        self.models = [xgb.XGBRegressor(random_state=s, **self.params).fit(X.values, y.values)
                       for s in range(self.n_seeds)]
        self.shrink = self.calibrate(X, y, horizon, n_folds) if horizon else 1.0
        return self

    def calibrate(self, X: pd.DataFrame, y: pd.Series, horizon: int, n_folds: int = 4) -> float:
        """Pente β ∈ [0, 1] de y sur les prédictions out-of-fold (CV par blocs purgés)."""
        n = len(y)
        Xv, yv = X.values, y.values
        oof = np.full(n, np.nan)
        for idx in np.array_split(np.arange(n), n_folds):
            lo, hi = idx[0] - horizon, idx[-1] + horizon
            train = np.r_[0:max(lo, 0), min(hi + 1, n):n]
            if len(train) < 30:
                continue
            m = xgb.XGBRegressor(random_state=0, **self.params).fit(Xv[train], yv[train])
            oof[idx] = m.predict(Xv[idx])
        ok = np.isfinite(oof)
        p, t = oof[ok], yv[ok]
        denom = p @ p
        return float(np.clip((p @ t) / denom, 0.0, 1.0)) if denom > 0 else 0.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            return np.zeros(len(X))
        raw = np.mean([m.predict(X[self.features].values) for m in self.models], axis=0)
        return self.shrink * raw

    def shap_importance(self, X: pd.DataFrame) -> pd.Series:
        """Moyenne des |valeurs SHAP| (TreeSHAP natif XGBoost), moyennée sur l'ensemble."""
        dm = xgb.DMatrix(X[self.features].values)
        contrib = np.mean([np.abs(m.get_booster().predict(dm, pred_contribs=True))[:, :-1].mean(axis=0)
                           for m in self.models], axis=0)
        return pd.Series(contrib, index=self.features)

    def gain_importance(self) -> pd.Series:
        tot = pd.Series(0.0, index=self.features)
        for m in self.models:
            g = m.get_booster().get_score(importance_type="total_gain")
            tot += pd.Series({self.features[int(k[1:])]: v for k, v in g.items()}).reindex(self.features).fillna(0)
        return tot / tot.sum() if tot.sum() > 0 else tot


def residual_target(resid: pd.Series, horizon: int) -> pd.Series:
    """cible_j = e_{j+1} + … + e_{j+h} ; NaN pour les h dernières dates (non observées)."""
    return resid.rolling(horizon).sum().shift(-horizon)
