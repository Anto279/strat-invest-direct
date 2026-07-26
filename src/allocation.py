from scipy.optimize import minimize
import numpy as np
import pandas as pd

class MarkowitzOptimizer:
    def __init__(self, risk_free_rate=0.02):
        self.rf = risk_free_rate

    def optimize_portfolio(self, expected_returns, cov_matrix, current_weights=None, 
                           constraints_bounds=None, transaction_cost=0.002,
                           regime_method="sma", bear_probs=None, lambda_reg=0.05): 
        """
        Optimisation Max Sharpe avec Régularisation L2 (Ridge) pour éviter la sur-concentration.
        """
        n_assets = len(expected_returns)
        tickers = expected_returns.index
        
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        else:
            current_weights = np.array(current_weights)

        # --- DÉFINITION DE LA FONCTION OBJECTIVE ---
        def objective_function(weights):
            ret = weights.T @ expected_returns
            vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            turnover = np.sum(np.abs(weights - current_weights))
            cost_penalty = turnover * transaction_cost
            
            # Pénalité L2 pour forcer la diversification (Empêche le "Winner takes all")
            l2_penalty = lambda_reg * np.sum(weights**2)
            
            net_return = ret - cost_penalty
            rf_period = self.rf / 52
            
            if vol > 1e-6:
                sharpe = (net_return - rf_period) / vol
            else:
                sharpe = 0.0
                
            # On maximise le Sharpe, tout en minimisant la concentration des poids
            return -(sharpe - l2_penalty)

        cons = ({'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))})

        if constraints_bounds is None:
            bounds = tuple((-1, 1) for _ in range(n_assets))
        else:
            bounds = tuple(constraints_bounds)

        if np.sum(np.abs(current_weights)) > 0.9:
            init_guess = current_weights
        else:
            init_guess = np.array([1.0/n_assets] * n_assets)
            
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
        
        if regime_method.lower() == "hmm" and bear_probs is not None:
            for i, ticker in enumerate(tickers):
                w = final_weights[i]
                p_bear = bear_probs.get(ticker, 0.0)
                p_bear = max(0.0, min(1.0, p_bear))
                p_bull = 1.0 - p_bear
                
                if w > 0:
                    final_weights[i] = w * p_bull
                elif w < 0:
                    final_weights[i] = w * p_bear

        final_weights[np.abs(final_weights) < 1e-4] = 0
        return pd.Series(final_weights, index=tickers)

    def portfolio_performance(self, weights, expected_returns, cov_matrix):
        ret = weights.T @ expected_returns
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe = ret / vol if vol > 1e-6 else 0
        return ret, vol, sharpe