import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM

class Strategy:
    def __init__(self, data):
        self.data = data.copy()

    def define_market_regime(self, col_ma_short='SMA_50', col_ma_long='SMA_200'):
        required_cols = ['Close', col_ma_short, col_ma_long]
        for col in required_cols:
            if col not in self.data.columns:
                return self.data

        cond_bullish = (self.data['Close'] > self.data[col_ma_short]) & \
                       (self.data[col_ma_short] > self.data[col_ma_long])

        cond_bearish = (self.data['Close'] < self.data[col_ma_short]) & \
                       (self.data[col_ma_short] < self.data[col_ma_long])

        conditions = [cond_bullish, cond_bearish]
        choices = [1, -1] 
        self.data['Regime'] = np.select(conditions, choices, default=0)
        
        return self.data

    def get_current_bounds(self, tickers):
        bounds = []
        if 'Ticker' in self.data.index.names:
            last_regimes = self.data.groupby(level='Ticker')['Regime'].last()
        elif 'Ticker' in self.data.columns:
            last_regimes = self.data.groupby('Ticker')['Regime'].last()
        else:
            last_val = self.data['Regime'].iloc[-1]
            last_regimes = pd.Series({tickers[0]: last_val})

        for t in tickers:
            regime = last_regimes.get(t, 0)
            if regime == 1:
                bounds.append((0, 1))
            elif regime == -1:
                bounds.append((-1, 0))
            else:
                bounds.append((0, 0))
        
        return bounds

    def get_signal_summary(self):
        if 'Regime' not in self.data.columns: return "Régime non défini."
        return self.data.groupby('Ticker')['Regime'].value_counts().unstack(fill_value=0)
    

class HMMStrategy:
    def __init__(self, data, n_states=2, train_window=52):
        self.data = data.copy()
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states, 
            covariance_type="full", 
            n_iter=100, 
            random_state=42
        )
        self.last_bear_prob = None

    def fit_predict(self, vol_window=12):
        df_feat = self.data.copy()
        
        if 'Log Returns' not in df_feat.columns:
            df_feat['Log Returns'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1))
            
        df_feat['HMM_Vol'] = df_feat['Log Returns'].rolling(window=vol_window).std() * 100
        df_feat['HMM_Ret'] = df_feat['Log Returns'] * 100
        
        df_clean = df_feat.dropna().copy()
        if df_clean.empty:
            return df_feat
            
        X = df_clean[['HMM_Ret', 'HMM_Vol']].values
        
        try:
            self.model.fit(X)
        except ValueError:
            df_clean['HMM_Regime'] = 0
            df_clean['Bear_Prob'] = 0.0
            return df_clean
        
        # --- CORRECTION QUANT : Identification par le Ratio de Sharpe ---
        ret_means = self.model.means_[:, 0]
        vol_means = self.model.means_[:, 1]
        
        # Sécurité division par zéro
        safe_vols = np.where(vol_means == 0, 1e-6, vol_means)
        sharpe_ratios = ret_means / safe_vols
        
        # L'état baissier est celui avec le pire rendement ajusté au risque
        bear_state = np.argmin(sharpe_ratios) 
        
        hidden_states = self.model.predict(X)
        probs = self.model.predict_proba(X)
        bear_probs = probs[:, bear_state]
        
        df_clean['HMM_Regime'] = np.where(hidden_states == bear_state, -1, 1)
        df_clean['Bear_Prob'] = bear_probs
        self.last_bear_prob = bear_probs[-1]

        return df_clean
    
    @staticmethod
    def get_bounds_from_probabilities(tickers, probs_dict, threshold=0.70):
        bounds = []
        for t in tickers:
            p_bear = probs_dict.get(t, 0.5)
            if p_bear > threshold:
                bounds.append((-1, 0))
            elif p_bear < (1.0 - threshold):
                bounds.append((0, 1))
            else:
                bounds.append((0, 0))
        return tuple(bounds)