"""
Configuration centrale du projet : univers d'investissement, benchmark et
hyper-paramètres de la pipeline.

Tous les paramètres qui influencent un résultat chiffré sont regroupés ici,
dans un objet immuable, afin que chaque backtest soit entièrement décrit par
sa configuration (reproductibilité + clé de cache déterministe).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Univers d'investissement
# ---------------------------------------------------------------------------
# Construction : une poche par prime de risque structurellement distincte
# (actions développées US / hors US / émergentes, duration intermédiaire et
# longue, or, matières premières, immobilier coté). Les doublons statistiques
# de l'ancienne version (QQQ, ρ ≈ 0.92 avec SPY ; LQD, ρ ≈ 0.8 avec IEF et
# exposé au même facteur crédit/actions) ont été retirés : ils augmentaient le
# conditionnement de la matrice de covariance sans ajouter de facteur.
# Historique commun disponible : février 2006 (lancement de DBC).

UNIVERSE: dict[str, dict[str, str]] = {
    "SPY": {"name": "S&P 500", "asset_class": "Actions", "role": "Prime de risque actions US (large caps)"},
    "EFA": {"name": "MSCI EAFE", "asset_class": "Actions", "role": "Actions développées hors US (Europe, Japon)"},
    "EEM": {"name": "MSCI Emerging Markets", "asset_class": "Actions", "role": "Actions émergentes"},
    "IEF": {"name": "US Treasuries 7-10 ans", "asset_class": "Taux", "role": "Duration intermédiaire, amortisseur"},
    "TLT": {"name": "US Treasuries 20+ ans", "asset_class": "Taux", "role": "Duration longue, couverture déflationniste"},
    "GLD": {"name": "Or physique", "asset_class": "Réels", "role": "Couverture monétaire / risque extrême"},
    "DBC": {"name": "Matières premières (indice large)", "asset_class": "Réels", "role": "Couverture inflation"},
    "VNQ": {"name": "Immobilier coté US (REITs)", "asset_class": "Réels", "role": "Prime immobilière, sensible aux taux"},
}
TICKERS: list[str] = list(UNIVERSE)

# Benchmark principal : portefeuille 60/40 (SPY / IEF) rééquilibré toutes les
# 4 semaines, frais inclus. Benchmark secondaire : 1/N sur l'univers.
BENCHMARK_WEIGHTS: dict[str, float] = {"SPY": 0.60, "IEF": 0.40}
BENCHMARK_NAME = "60/40 (SPY / IEF)"
EQUAL_WEIGHT_NAME = "1/N univers"

# Variables macro utilisées comme features (pas comme actifs investissables).
MACRO_TICKERS: dict[str, str] = {"VIX": "^VIX", "TNX": "^TNX", "DXY": "DX-Y.NYB"}
# Taux sans risque : rendement des T-Bills 13 semaines (en %, annualisé).
RISK_FREE_TICKER = "^IRX"

DATA_START = "2005-01-01"


# ---------------------------------------------------------------------------
# Paramètres de modélisation (walk-forward)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 4                  # horizon de prévision (semaines) = période de détention
    train_window: int = 260           # fenêtre glissante d'estimation (5 ans hebdomadaires)
    arima_max_p: int = 3
    arima_max_q: int = 3
    arima_reselect_every: int = 52    # re-sélection de l'ordre (p, q) une fois par an
    arima_criterion: str = "BIC"      # critère de sélection (BIC : convergent, parcimonieux)
    garch_dist: str = "t"             # innovations Student-t (queues épaisses)
    xgb_refit_every: int = 4          # ré-entraînement XGBoost toutes les 4 semaines
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 2
    xgb_learning_rate: float = 0.01
    xgb_n_seeds: int = 3              # ensemble de graines (réduction de variance)
    sma_windows: tuple[int, ...] = (30, 40, 52)  # filtres de tendance (semaines)
    corr_window: int = 104            # fenêtre d'estimation des corrélations (2 ans)

    def key(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class PortfolioConfig:
    risk_aversion: float = 6.0        # γ de l'utilité moyenne-variance
    view_weight: float = 0.5          # κ : poids des prévisions ARIMA-XGBoost vs prior
    prior_sharpe: float = 0.30        # Sharpe annuel du prior d'équilibre (Sharpe constant)
    max_weight: float = 0.30          # borne haute par actif (avant filtre de tendance)
    vol_cap: float = 0.10             # volatilité ex-ante annuelle maximale (Safety-First)
    use_trend_filter: bool = True
    forecast_source: str = "hybrid_mu"  # "hybrid_mu" | "arima_mu" | "hist_mu"

    def key(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2011-01-01"
    rebalance_every: int = 4          # semaines entre deux rééquilibrages d'une tranche
    n_tranches: int = 4               # nombre de sous-portefeuilles décalés
    cost_bps: float = 10.0            # coût one-way (commission + demi-spread), en bps
    buffer: float = 0.05              # bande de non-transaction sur le turnover one-way
    execution_lag: int = 1            # semaines entre le signal et l'exécution

    def key(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]
