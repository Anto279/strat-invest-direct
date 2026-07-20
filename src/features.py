import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, data):
        # On suppose que data est indexé par [Date, Ticker] ou juste Date
        self.data = data.copy()

    def add_moving_average(self, window=50, title='SMA50'):
        """ Calcule la moyenne mobile par actif. """
        self.data[title] = self.data.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window=window).mean()
        )
        return self.data

    def add_rsi(self, window=14):
        """ Calcule le RSI par actif. """
        def _calc_rsi_series(series):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).ewm(span=window, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(span=window, adjust=False).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        self.data['RSI'] = self.data.groupby('Ticker')['Close'].transform(_calc_rsi_series)
        return self.data

    def add_lags(self, lags=[1, 2, 3]):
        """
        Ajoute les rendements passés comme features (Lagged Returns).
        Crucial pour que XGBoost capte le 'Momentum'.
        """
        # On s'assure d'avoir les Log Returns calculés
        if 'Log Returns' not in self.data.columns:
            self.data['Log Returns'] = self.data.groupby('Ticker')['Close'].transform(
                lambda x: np.log(x) - np.log(x.shift(1))
            )
            
        for lag in lags:
            col_name = f'Ret_Lag_{lag}'
            self.data[col_name] = self.data.groupby('Ticker')['Log Returns'].transform(
                lambda x: x.shift(lag)
            )
        return self.data
    
    def add_bollinger_bands(self, window=20, num_std=2):
        """
        Ajoute les Bandes de Bollinger.
        - BB_Pct (%B) : Position du prix par rapport aux bandes (0=Bas, 1=Haut).
        - BB_Width : Volatilité relative (Indispensable pour le Pétrole).
        """
        sma = self.data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window).mean())
        std = self.data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window).std())
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        # Position relative (Oscillateur borné 0-1)
        self.data['BB_Pct'] = (self.data['Close'] - lower) / (upper - lower)
        
        # Largeur des bandes (Indicateur de volatilité pure)
        # Si c'est bas = Squeeze (Explosion imminente) -> Très fort pour l'Or et le Pétrole
        self.data['BB_Width'] = (upper - lower) / sma
        
        return self.data

    def add_macd(self, fast=12, slow=26, signal=9): # fast: 12 weeks, slow: 26 weeks
        """
        Ajoute le MACD (Momentum).
        Très utile pour les indices (CAC40, SP500) pour confirmer la tendance.
        """
        ema_fast = self.data.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=fast, adjust=False).mean())
        ema_slow = self.data.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=slow, adjust=False).mean())
        
        self.data['MACD'] = ema_fast - ema_slow
        self.data['MACD_Signal'] = self.data.groupby('Ticker')['MACD'].transform(lambda x: x.ewm(span=signal, adjust=False).mean())
        
        # L'histogramme montre l'accélération du mouvement
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        
        return self.data
    
    def add_context(self, context_df, prefix='Macro'):
        """
        Fusionne des données macroéconomiques (ex: VIX, Taux) avec l'actif courant.
        context_df doit avoir un index Date et une colonne 'Close'.
        """
        # 1. Préparation des données Macro (Calcul des rendements AVANT la fusion)
        # On travaille sur une copie pour ne pas modifier l'original
        ctx = context_df.copy()
        
        # On s'assure que l'index est Timezone-naive pour le mapping
        ctx.index = ctx.index.tz_localize(None)
        
        # On gère les doublons éventuels dans le contexte (garder le premier)
        ctx = ctx[~ctx.index.duplicated(keep='first')]
        
        # Création des séries propres
        # Niveau (Prix/Indice)
        s_close = ctx['Close']
        # Variation (Rendement) - Calculé proprement sur la série temporelle unique
        s_ret = ctx['Close'].pct_change()
        
        # 2. Préparation des Dates Cibles (Celles de votre DataFrame principal)
        if hasattr(self.data.index, 'levels'): # MultiIndex (Date, Ticker)
            target_dates = self.data.index.get_level_values('Date').tz_localize(None)
        else: # Index Simple (Date)
            target_dates = self.data.index.tz_localize(None)
            
        # 3. Mapping (Fusion)
        # On utilise .map() qui est très rapide et gère l'alignement automatiquement
        self.data[f"{prefix}_Close"] = target_dates.map(s_close)
        self.data[f"{prefix}_Ret"] = target_dates.map(s_ret)
        
        return self.data