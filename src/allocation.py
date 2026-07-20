from scipy.optimize import minimize
import numpy as np
import pandas as pd

class MarkowitzOptimizer:
    def __init__(self, risk_free_rate=0.02):
        self.rf = risk_free_rate

    def optimize_portfolio(self, expected_returns, cov_matrix, current_weights=None, 
                           constraints_bounds=None, transaction_cost=0.002,
                           regime_method="sma", bear_probs=None): 
        """
        Optimisation Max Sharpe Ratio avec gestion hybride des régimes (SMA ou HMM).
        
        Args:
            expected_returns (pd.Series): Rendements espérés (mu).
            cov_matrix (pd.DataFrame): Matrice de covariance (Sigma).
            current_weights (np.array): Poids actuels (pour pénalité turnover).
            constraints_bounds (list): Bornes dynamiques (Utilisé par SMA pour Hard limit).
            transaction_cost (float): Coûts de transaction.
            regime_method (str): 'sma' (binaire) ou 'hmm' (probabiliste).
            Args:
            bear_probs (dict ou pd.Series): Probabilité de Bear Market spécifique à chaque actif.
                                            Ex: {'AAPL': 0.10, 'TSLA': 0.80}
        """
        n_assets = len(expected_returns)
        tickers = expected_returns.index
        
        # 0. Gestion du portefeuille actuel
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        else:
            current_weights = np.array(current_weights)

        # --- DÉFINITION DE LA FONCTION OBJECTIVE (Inchangée) ---
        def objective_function(weights):
            ret = weights.T @ expected_returns
            vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # Pénalité de Turnover
            turnover = np.sum(np.abs(weights - current_weights))
            cost_penalty = turnover * transaction_cost
            
            # Sharpe Ratio Net
            net_return = ret - cost_penalty
            rf_period = self.rf / 52
            
            if vol > 1e-6:
                sharpe = (net_return - rf_period) / vol
            else:
                sharpe = 0.0
                
            return -sharpe

        # --- CONTRAINTES & BORNES ---
        # Contrainte : Somme(|w|) <= 1 (Max 100% investi, le reste est Cash)
        cons = ({'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))})

        # Gestion des Bornes selon la méthode
        if constraints_bounds is None:
            # Par défaut : Long/Short autorisé (-1, 1) ou Long Only (0, 1) selon ta préf.
            bounds = tuple((-1, 1) for _ in range(n_assets))
        else:
            # Si méthode SMA : Les bornes sont strictes (ex: (0,0) si tendance baissière)
            # Si méthode HMM : On peut passer des bornes "ouvertes" et laisser le HMM gérer l'expo globale
            bounds = tuple(constraints_bounds)

        # --- POINT DE DÉPART ---
        if np.sum(np.abs(current_weights)) > 0.9:
            init_guess = current_weights
        else:
            # Fallback
            init_guess = np.array([1.0/n_assets] * n_assets)
            
        # --- 2. OPTIMISATION CŒUR (Markowitz) ---
        # Cette étape trouve la "Structure Optimale" (Corrélation/Diversification)
        result = minimize(
            fun=objective_function,
            x0=init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            tol=1e-6
        )

        if not result.success:
            final_weights = init_guess
        else:
            final_weights = result.x.copy()

        # --- 2. GESTION DES RÉGIMES (ACTIF PAR ACTIF) ---
        
        # CAS HMM : On ajuste la taille de la position selon la proba Bear de CHAQUE actif
        if regime_method.lower() == "hmm" and bear_probs is not None:
            
            # On parcourt chaque actif par son index numérique
            for i, ticker in enumerate(tickers):
                w = final_weights[i]
                
                # On récupère la proba spécifique à cet actif (0.0 par défaut si introuvable)
                p_bear = bear_probs.get(ticker, 0.0)
                
                # Sécurité bornes
                p_bear = max(0.0, min(1.0, p_bear))
                p_bull = 1.0 - p_bear
                
                if w > 0:
                    # LONG : On réduit si l'actif est en régime Bear
                    # Si P(Bear) de l'actif = 0.8 -> On ne garde que 20% de la position
                    final_weights[i] = w * p_bull
                    
                elif w < 0:
                    # SHORT : On réduit si l'actif est en régime Bull
                    # Si P(Bear) de l'actif = 0.1 (donc Bull) -> On ne garde que 10% du Short
                    final_weights[i] = w * p_bear
                    
            # Le "delta" non investi devient implicitement du Cash

        # Nettoyage
        final_weights[np.abs(final_weights) < 1e-4] = 0
        
        return pd.Series(final_weights, index=tickers)

    def portfolio_performance(self, weights, expected_returns, cov_matrix):
        ret = weights.T @ expected_returns
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe = ret / vol if vol > 1e-6 else 0
        return ret, vol, sharpe