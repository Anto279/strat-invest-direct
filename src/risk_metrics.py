import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

def calculate_gdr_metrics(returns_series, confidence_level=0.95):
    """
    Calcule les métriques de risque avancées basées sur le cours GDR.
    
    Args:
        returns_series (pd.Series): Série des rendements (Log Returns ou Net Returns).
        confidence_level (float): Niveau de confiance (ex: 0.95 pour 95%).
        
    Returns:
        dict: Dictionnaire contenant VaR Hist, VaR Param, CVaR, Skewness, Kurtosis.
    """
    # On travaille avec les pertes (returns négatifs)
    # alpha est le seuil de queue (ex: 0.05 pour 95% de confiance)
    alpha = 1 - confidence_level
    
    mu = returns_series.mean()
    sigma = returns_series.std()
    
    # --- 1. VaR Historique (Non-Paramétrique) ---
    # On cherche le percentile exact dans les données passées
    # Si alpha = 5%, on cherche la valeur où 5% des données sont inférieures
    var_hist = np.percentile(returns_series, alpha * 100)
    
    # --- 2. VaR Paramétrique (Gaussienne / Normale) ---
    # Formule du cours : mu + z_alpha * sigma
    # ppf = Percent Point Function (Inverse de la CDF)
    z_score = norm.ppf(alpha) 
    var_param = mu + (z_score * sigma)
    
    # --- 3. VaR Cornish-Fisher (Modifiée) ---
    # Ajuste la VaR pour les distributions non-normales (Skewness & Kurtosis)
    # C'est souvent plus précis pour les Hedge Funds / Stratégies actives
    s = skew(returns_series)
    k = kurtosis(returns_series) # Fisher kurtosis (Normal = 0)
    
    # Approximation de Cornish-Fisher pour le quantile ajusté
    z_cf = z_score + (1/6)*(z_score**2 - 1)*s + (1/24)*(z_score**3 - 3*z_score)*k - (1/36)*(2*z_score**3 - 5*z_score)*(s**2)
    var_cf = mu + (z_cf * sigma)

    # --- 4. Expected Shortfall (CVaR) ---
    # Moyenne des rendements qui sont inférieurs à la VaR Historique
    # C'est l'espérance conditionnelle de perte
    cvar_hist = returns_series[returns_series <= var_hist].mean()
    
    return {
        "VaR Historique (95%)": var_hist,
        "VaR Paramétrique (95%)": var_param,
        "VaR Cornish-Fisher (95%)": var_cf,
        "CVaR / Expected Shortfall (95%)": cvar_hist,
        "Skewness": s,
        "Kurtosis": k
    }