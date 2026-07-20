import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM

class Strategy:
    def __init__(self, data):
        """
        Initialise la stratégie avec un DataFrame de données.
        """
        self.data = data.copy()

    def define_market_regime(self, col_ma_short='SMA_50', col_ma_long='SMA_200'):
        """
        Définit le régime de marché.
        Pour des données Hebdomadaires, on utilise souvent 50 (un an) et 200 (4 ans).
        
        Règles :
        - Haussier (1) : Prix > MM_Court > MM_Long
        - Baissier (-1): Prix < MM_Court < MM_Long
        - Neutre (0)   : Autres cas
        """
        # Vérification
        required_cols = ['Close', col_ma_short, col_ma_long]
        for col in required_cols:
            if col not in self.data.columns:
                # Si colonnes absentes, on ne peut pas calculer, on renvoie tel quel
                return self.data

        # Conditions Vectorisées
        cond_bullish = (self.data['Close'] > self.data[col_ma_short]) & \
                       (self.data[col_ma_short] > self.data[col_ma_long])

        cond_bearish = (self.data['Close'] < self.data[col_ma_short]) & \
                       (self.data[col_ma_short] < self.data[col_ma_long])

        conditions = [cond_bullish, cond_bearish]
        choices = [1, -1] # 1 = Long Only, -1 = Short Only
        
        self.data['Regime'] = np.select(conditions, choices, default=0) # 0 = Cash
        
        return self.data

    def get_current_bounds(self, tickers):
        """
        Retourne les bornes pour l'optimiseur (Constraints) basées sur le DERNIER régime connu.
        Ordre aligné avec la liste 'tickers' fournie.
        
        Returns:
            list of tuple: [(min, max), (min, max), ...]
        """
        bounds = []
        
        # On prend la dernière ligne disponible pour chaque ticker
        # GroupBy Ticker -> Last Value
        if 'Ticker' in self.data.index.names:
            last_regimes = self.data.groupby(level='Ticker')['Regime'].last()
        elif 'Ticker' in self.data.columns:
            last_regimes = self.data.groupby('Ticker')['Regime'].last()
        else:
            # Cas Single Asset
            last_val = self.data['Regime'].iloc[-1]
            last_regimes = pd.Series({tickers[0]: last_val})

        for t in tickers:
            # 0 par défaut si ticker non trouvé
            regime = last_regimes.get(t, 0)
            
            if regime == 1:
                bounds.append((0, 1))   # Haussier -> Achat Uniquement
            elif regime == -1:
                bounds.append((-1, 0))  # Baissier -> Vente à découvert Uniquement
            else:
                bounds.append((0, 0))   # Neutre -> Cash (Pas de position)
        
        return bounds

    def get_signal_summary(self):
        if 'Regime' not in self.data.columns: return "Régime non défini."
        return self.data.groupby('Ticker')['Regime'].value_counts().unstack(fill_value=0)
    

class HMMStrategy:
    def __init__(self, data, n_states=2, train_window=52):
        """
        Stratégie basée sur les Régimes de Marché (HMM).
        Algorithme : Baum-Welch pour l'entraînement, puis Viterbi pour la prédiction.
        Args:
            data (pd.DataFrame): Doit contenir 'Log Returns' (et 'Close' pour le calcul initial).
            n_states (int): Nombre d'états (2 = Bull/Bear).
            train_window (int): Non utilisé ici pour le fit global, mais utile pour le Walk-Forward.
        """
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
        """
        Entraîne le HMM et détecte les régimes.
        Args:
            vol_window (int): Fenêtre pour la volatilité roulante. 
                              En Hebdo, 12 = 1 trimestre.
        Returns:
            pd.DataFrame: DataFrame avec la colonne 'HMM_Regime' (-1: Bear, 1: Bull)
        """
        # 1. Vérification / Création des Features
        df_feat = self.data.copy()
        
        # Si 'Log Returns' n'existe pas, on le crée
        if 'Log Returns' not in df_feat.columns:
            df_feat['Log Returns'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1))
            
        # Feature 2 : Volatilité Roulante (Indicateur clé pour le HMM)
        # On multiplie par 100 pour la stabilité numérique du HMM (évite les trop petits chiffres)
        df_feat['HMM_Vol'] = df_feat['Log Returns'].rolling(window=vol_window).std() * 100
        df_feat['HMM_Ret'] = df_feat['Log Returns'] * 100
        
        # Nettoyage des NaN (dus au rolling et au shift)
        df_clean = df_feat.dropna().copy()
        
        if df_clean.empty:
            return df_feat # Retourne vide si pas assez de données
            
        # 2. Préparation Matrice X (Features)
        X = df_clean[['HMM_Ret', 'HMM_Vol']].values
        
        # 3. Entraînement
        try:
            self.model.fit(X)
        except ValueError:
            df_clean['HMM_Regime'] = 0
            df_clean['Bear_Prob'] = 0.0 # Fallback
            return df_clean
        
        # 4. Identification des États (Quel est le Bear ?)
        vol_means = self.model.means_[:, 1]
        bear_state = np.argmax(vol_means) # L'état avec la plus grosse Vol
        
        # 5. Prédiction des états (Séquence)
        hidden_states = self.model.predict(X)
        
        # --- NOUVEAUTÉ : Probabilités Postérieures ---
        # predict_proba renvoie une matrice (n_samples, n_states)
        probs = self.model.predict_proba(X)
        
        # On extrait la probabilité d'être dans l'état "Bear"
        bear_probs = probs[:, bear_state]
        
        # 6. Stockage dans le DataFrame
        df_clean['HMM_Regime'] = np.where(hidden_states == bear_state, -1, 1)
        df_clean['Bear_Prob'] = bear_probs # La colonne cruciale pour l'allocation

        self.last_bear_prob = bear_probs[-1]

        return df_clean
    
    @staticmethod
    def get_bounds_from_probabilities(tickers, probs_dict, threshold=0.70):
        """
        Convertit un dictionnaire de probabilités en bornes strictes (Hard Bounds).
        
        Règles :
        - P(Bear) > 70%         -> Vente à découvert uniquement (-1, 0)
        - P(Bull) > 70%         -> Achat uniquement (0, 1)
          (soit P(Bear) < 30%)
        - Sinon (Incertitude)   -> Cash (0, 0)
        
        Args:
            tickers (list): Liste des tickers pour garantir l'ordre.
            probs_dict (dict): { 'AAPL': 0.12, 'TSLA': 0.85, ... }
            threshold (float): Seuil de conviction (ex: 0.70).
            
        Returns:
            tuple: Tuple de tuples de bornes ((min, max), ...)
        """
        bounds = []
        
        for t in tickers:
            # On récupère la proba, par défaut 0.5 (Incertain) si absente
            p_bear = probs_dict.get(t, 0.5)
            
            if p_bear > threshold:
                # Conviction Baissière -> Short
                bounds.append((-1, 0))
            elif p_bear < (1.0 - threshold):
                # Conviction Haussière -> Long
                bounds.append((0, 1))
            else:
                # Zone Grise -> Cash
                bounds.append((0, 0))
                
        return tuple(bounds)