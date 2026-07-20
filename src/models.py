import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from pmdarima.arima import ARIMA
from arch import arch_model
import xgboost as xgb
from typing import Tuple

# --- CONFIGURATION DES CLUSTERS D'ACTIFS ---
ASSETS_CONFIG = {
    # Indices
    '^GSPC': 'Index', '^IXIC': 'Index', '^FCHI': 'Index', '^DJI': 'Index', 
    '^GDAXI': 'Index', '^N225': 'Index', '^FTSE': 'Index', '^HSI': 'Index',
    'SPY': 'Index', 'QQQ': 'Index', 'IWM': 'Index', '^RUT': 'Index', 
    '^STOXX50E': 'Index', '^BVSP': 'Index',
    
    # Commodities
    'GLD': 'Commodities', 'USO': 'Commodities', 'SLV': 'Commodities', 
    'PPLT': 'Commodities', 'UNG': 'Commodities', 'DBA': 'Commodities', 
    'DBC': 'Commodities', 'COPX': 'Commodities', 'CORN': 'Commodities',
    'PALL': 'Commodities', 'WOOD': 'Commodities',
    
    # Bonds
    'TLT': 'Bonds', 'IEF': 'Bonds', 'SHY': 'Bonds', 
    'LQD': 'Bonds', 'HYG': 'Bonds', 'BND': 'Bonds', 'TIP': 'Bonds',
    'AGG': 'Bonds', '^TNX': 'Bonds',
        
    # Forex
    'EURUSD=X': 'Forex', 'GBPUSD=X': 'Forex', 'JPY=X': 'Forex', 
    'AUDUSD=X': 'Forex', 'USDCAD=X': 'Forex', 'CHF=X': 'Forex', 
    'NZDUSD=X': 'Forex', 'AUDJPY=X': 'Forex', 'EURGBP=X': 'Forex'
}

class ModelPreparator:
    def __init__(self, data):
        self.data = data.sort_index().copy()

    def prepare_data(self):
        """ Prépare les features et la target. """
        # 1. Rendements Simples
        self.data['Simple Returns'] = self.data.groupby(level='Ticker')['Close'].transform(
            lambda x: x.pct_change()
        )

        # 2. Rendements Logarithmiques
        self.data['Log Returns'] = self.data.groupby(level='Ticker')['Close'].transform(
            lambda x: np.log(x) - np.log(x.shift(1))
        )

        # 3. Target (J+1)
        self.data['Target'] = self.data.groupby(level='Ticker')['Log Returns'].transform(
            lambda x: x.shift(-1)
        )
        

        self.data = self.data.dropna()
        return self.data

    
    def train_test_split(self, train_size=0.8):
        dates = self.data.index.get_level_values('Date').unique().sort_values()
        split_idx = int(len(dates) * train_size)
        split_date = dates[split_idx]

        train_data = self.data[self.data.index.get_level_values('Date') < split_date]
        test_data = self.data[self.data.index.get_level_values('Date') >= split_date]

        return train_data, test_data
    
    def train_test_split_by_date(self, split_date_str):
        """
        Sépare les données en Train et Test basé sur une date de coupure.
        Tout ce qui est AVANT la date va dans le Train.
        Tout ce qui est APRÈS (ou égal) va dans le Test.
        
        Args:
            split_date_str (str): La date de début du test (ex: "2023-01-01")
        """
        # 1. Conversion de la string en Timestamp pour comparaison
        split_dt = pd.to_datetime(split_date_str)
        
        # 2. Création des masques
        # On récupère l'index des dates
        dates = self.data.index.get_level_values('Date')
        
        # Train : Strictement avant la date
        train_data = self.data[dates < split_dt].copy()
        
        # Test : À partir de la date (inclus)
        test_data = self.data[dates >= split_dt].copy()

        # Petite vérification de sécurité
        if train_data.empty or test_data.empty:
            print(f"⚠️ Attention : Le split à la date {split_date_str} a généré un ensemble vide.")
            print(f"   Train size: {len(train_data)}, Test size: {len(test_data)}")

        return train_data, test_data
    
    
    def get_feature_names(self, data):
        cols_to_drop = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Date', 'Simple Returns', 'Log Returns', 'Log_Price',
                        'SMA_5', 'SMA_13', 'SMA_20', 'SMA_26', 'SMA_50', 'SMA_100', 'SMA_200', 'Regime']
        potential = [c for c in data.columns if c not in cols_to_drop]
        return data[potential].select_dtypes(include=[np.number]).columns.tolist()

    def get_X_y(self, train_data, test_data):
        cols_to_drop = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Date', 'Simple Returns', 'Log Returns', 'Log_Price',
                        'SMA_5', 'SMA_13', 'SMA_20', 'SMA_26', 'SMA_50', 'SMA_100', 'SMA_200', 'Regime']
        existing_drop = [c for c in cols_to_drop if c in train_data.columns]

        X_train = train_data.drop(columns=existing_drop)
        y_train = train_data['Target']
        X_test = test_data.drop(columns=existing_drop)
        y_test = test_data['Target']

        scaler = StandardScaler()
        X_train_numeric = X_train.select_dtypes(include=[np.number])
        X_test_numeric = X_test.select_dtypes(include=[np.number])

        if X_train_numeric.empty:
            raise ValueError("Aucune feature numérique trouvée.")

        X_train_scaled = scaler.fit_transform(X_train_numeric)
        X_test_scaled = scaler.transform(X_test_numeric)

        return X_train_scaled, y_train, X_test_scaled, y_test, scaler
    
    def reconstruct_prices(self, predictions, index):
        """ Reconstruit les prix à partir d'un seul vecteur de prédiction (ex: Log Returns totaux). """
        try:
            current_prices = self.data.loc[index, 'Close']
            actual_log_returns = self.data.loc[index, 'Target']
        except KeyError:
            current_prices = self.data.loc[index]['Close'] if 'Close' not in index.to_frame().columns else index
            actual_log_returns = self.data.loc[index]['Target']

        predicted_prices = current_prices * np.exp(predictions)
        actual_future_prices = current_prices * np.exp(actual_log_returns)

        results = pd.DataFrame({
            'Close_t': current_prices,
            'Predicted_Close_t+1': predicted_prices,
            'Actual_Close_t+1': actual_future_prices
        }, index=index)
        return results

    # --- NOUVEAU : RECONSTRUCTION HYBRIDE (ARIMA + XGBoost) ---
    def reconstruct_hybrid_prices(self, arima_predictions, xgb_residual_predictions, index):
        """
        Reconstruit les prix en combinant :
        1. La prédiction de Tendance (ARIMA)
        2. La prédiction des Résidus (XGBoost)
        
        Total Log Return = ARIMA_Pred + XGB_Resid_Pred
        """
        # Somme des deux composantes pour obtenir le rendement total prédit
        total_predicted_log_returns = arima_predictions + xgb_residual_predictions
        
        # On réutilise la logique de reconstruction standard avec ce total
        return self.reconstruct_prices(total_predicted_log_returns, index)
    # -----------------------------------------------------------


class XGBoostModeler:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': 'reg:squarederror', # On va le surcharger lors du train
            'n_jobs': -1,
            'tree_method': 'hist', # Plus rapide
            'random_state': random_state
        }
        self.model = None

    def _custom_directional_loss(self, preds: np.ndarray, target) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fonction objective personnalisée (Gradient & Hessian).
        Pénalise lourdement les erreurs de signe (Direction).
        
        Si (pred * true) < 0  => Erreur de direction => Pénalité x10
        Sinon => Erreur standard (MSE)
        """
        # --- CORRECTION DE COMPATIBILITÉ ---
        # Vérifie si 'target' est un objet DMatrix (XGBoost interne) ou directement un array (Scikit-Learn wrapper)
        if hasattr(target, 'get_label'):
            labels = target.get_label()
        else:
            labels = target
        # -----------------------------------
        
        # Le Gradient (dérivée première)
        grad = preds - labels
        
        # Pénalité directionnelle : on amplifie le gradient si le signe est faux
        # On évite la division par zéro ou les signes nuls
        penalty_mask = (preds * labels) < 0
        grad[penalty_mask] *= 10.0  # Facteur de pénalité arbitraire (tunable)
        
        # Le Hessien (dérivée seconde)
        hess = np.ones_like(preds)
        hess[penalty_mask] *= 10.0
        
        return grad, hess

    def train(self, X_train, y_train, use_custom_loss=True):
        print(f"--- Entraînement XGBoost ({'Custom Loss' if use_custom_loss else 'MSE'}) ---")
        
        # 1. On fait une copie pour ne pas modifier les params originaux
        train_params = self.params.copy()
        
        if use_custom_loss:
            # CORRECTION : On supprime 'objective' du dictionnaire car on va le passer manuellement
            if 'objective' in train_params:
                del train_params['objective']
            
            self.model = xgb.XGBRegressor(
                **train_params,
                objective=self._custom_directional_loss # On injecte la fonction ici
            )
        else:
            self.model = xgb.XGBRegressor(**train_params)
            
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle avec des métriques classiques ET directionnelles.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        predictions = self.predict(X_test)
        
        # Métriques classiques
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Métrique Scientifique Importante : Précision Directionnelle
        # Quel pourcentage de fois le modèle a-t-il le bon signe ?
        # On évite le cas où prediction est exactement 0 avec une petite epsilon ou une gestion de signe
        correct_direction = np.sign(predictions) == np.sign(y_test)
        dir_accuracy = np.mean(correct_direction)
        
        return {
            'RMSE': rmse, 
            'MAE': mae, 
            'R2': r2, 
            'Directional_Accuracy': dir_accuracy, # Très important à afficher dans Streamlit
            'Predictions': predictions
        }
    
    def get_feature_importance(self, feature_names):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if len(feature_names) != len(importances):
                return None
            
            df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            return df.sort_values(by='Importance', ascending=True)
        return None


class StatisticalModeler:
    """
    Gère ARIMA-GARCH en mode 'Numérique pur' pour éviter les bugs de dates Pandas 2.0+.
    Supporte maintenant le mode 'Residuals' pour l'approche hybride.
    """
    def __init__(self, returns_series):
        # 1. Nettoyage et Tri
        self.returns = returns_series.dropna().sort_index()
        
        # 2. Scaling (x100) pour aider la convergence mathématique
        self.scale_factor = 100.0
        self.scaled_returns_series = self.returns * self.scale_factor
        
        # 3. Extraction des valeurs numpy pures
        self.train_values = self.scaled_returns_series.values
        
        self.arima_model = None
        self.garch_model = None
        self.garch_results = None
        self.residuals = None

    def check_stationarity(self):
        if len(self.train_values) < 10: return False
        result = adfuller(self.train_values)
        return result[1] < 0.05


    def fit_arima(self, seasonal=False, fixed_order=None):
        """
        Entraîne un modèle ARIMA.
        Si fixed_order est fourni (ex: (7, 0, 6)), on force ce modèle.
        Sinon, on lance auto_arima pour chercher le meilleur.
        """
        print("--- Fit ARIMA (Mode Numérique) ---")
        try:
            if fixed_order is not None:
                # CAS 1 : On force l'ordre manuel
                print(f"🔒 Ordre ARIMA forcé : {fixed_order}")
                # On utilise la classe ARIMA directe (plus rapide qu'auto_arima contraint)
                self.arima_model = ARIMA(
                    order=fixed_order, 
                    seasonal_order=None if not seasonal else (1, 0, 1, 12), # Exemple simple si saisonnier
                    suppress_warnings=True
                )
                self.arima_model.fit(self.train_values)
                
            else:
                # CAS 2 : Recherche Automatique (Auto-Arima)
                self.arima_model = auto_arima(
                    self.train_values, 
                    start_p=1, start_q=1,
                    max_p=5, max_q=5,
                    d=None,          
                    seasonal=seasonal,
                    trace=False,       
                    error_action='ignore',  
                    suppress_warnings=True, 
                    stepwise=True
                )
            
            # Affichage de confirmation
            print(f"✅ Modèle entraîné : {self.arima_model.order}")
            
            # Calcul des résidus pour la suite (GARCH / XGBoost)
            self.residuals = self.arima_model.resid()
            return self.arima_model

        except Exception as e:
            print(f"Erreur ARIMA: {e}")
            return None

    def fit_garch(self, p=1, q=1):
        if self.residuals is None: return None
        print("--- Fit GARCH (Mode Numérique) ---")
        try:
            garch = arch_model(self.residuals, vol='Garch', p=p, q=q, mean='Zero', dist='Normal')
            self.garch_results = garch.fit(disp='off')
            return self.garch_results
        except Exception as e:
            print(f"Erreur GARCH: {e}")
            return None

    def predict_next_step(self):
        """ Prédiction 1 pas en avant (dé-scalée) """
        if self.arima_model is None or self.garch_results is None:
            return 0.0, 0.0

        # Predict sur 1 période
        pred_scaled = self.arima_model.predict(n_periods=1)
        val_scaled = pred_scaled[0] if isinstance(pred_scaled, (list, np.ndarray)) else pred_scaled
        
        # Volatilité
        fcast = self.garch_results.forecast(horizon=1)
        
        # Gestion hybride DataFrame/Numpy pour la variance
        if hasattr(fcast.variance, 'iloc'):
            var_scaled = fcast.variance.iloc[-1, 0]
        else:
            if fcast.variance.ndim > 1:
                var_scaled = fcast.variance[-1, 0]
            else:
                var_scaled = fcast.variance[-1]
            
        vol_scaled = np.sqrt(var_scaled)
        
        return val_scaled / self.scale_factor, vol_scaled / self.scale_factor