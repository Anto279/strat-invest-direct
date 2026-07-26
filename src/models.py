import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from pmdarima.arima import ARIMA
from arch import arch_model
import xgboost as xgb
from typing import Tuple

ASSETS_CONFIG = {
    '^GSPC': 'Index', '^IXIC': 'Index', '^FCHI': 'Index', '^DJI': 'Index', 
    '^GDAXI': 'Index', '^N225': 'Index', '^FTSE': 'Index', '^HSI': 'Index',
    'SPY': 'Index', 'QQQ': 'Index', 'IWM': 'Index', '^RUT': 'Index', 
    '^STOXX50E': 'Index', '^BVSP': 'Index',
    'GLD': 'Commodities', 'USO': 'Commodities', 'SLV': 'Commodities', 
    'PPLT': 'Commodities', 'UNG': 'Commodities', 'DBA': 'Commodities', 
    'DBC': 'Commodities', 'COPX': 'Commodities', 'CORN': 'Commodities',
    'PALL': 'Commodities', 'WOOD': 'Commodities',
    'TLT': 'Bonds', 'IEF': 'Bonds', 'SHY': 'Bonds', 
    'LQD': 'Bonds', 'HYG': 'Bonds', 'BND': 'Bonds', 'TIP': 'Bonds',
    'AGG': 'Bonds', '^TNX': 'Bonds',
    'EURUSD=X': 'Forex', 'GBPUSD=X': 'Forex', 'JPY=X': 'Forex', 
    'AUDUSD=X': 'Forex', 'USDCAD=X': 'Forex', 'CHF=X': 'Forex', 
    'NZDUSD=X': 'Forex', 'AUDJPY=X': 'Forex', 'EURGBP=X': 'Forex'
}

class ModelPreparator:
    def __init__(self, data):
        self.data = data.sort_index().copy()

    def prepare_data(self):
        self.data['Simple Returns'] = self.data.groupby(level='Ticker')['Close'].transform(
            lambda x: x.pct_change()
        )
        self.data['Log Returns'] = self.data.groupby(level='Ticker')['Close'].transform(
            lambda x: np.log(x) - np.log(x.shift(1))
        )
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
        split_dt = pd.to_datetime(split_date_str)
        dates = self.data.index.get_level_values('Date')
        train_data = self.data[dates < split_dt].copy()
        test_data = self.data[dates >= split_dt].copy()
        return train_data, test_data
    
    def get_feature_names(self, data):
        cols_to_drop = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Date', 'Simple Returns', 'Log Returns', 'Log_Price',
                        'SMA_5', 'SMA_13', 'SMA_20', 'SMA_26', 'SMA_50', 'SMA_100', 'SMA_200', 'Regime']
        potential = [c for c in data.columns if c not in cols_to_drop]
        return data[potential].select_dtypes(include=[np.number]).columns.tolist()

    def get_X_y(self, train_data, test_data, rolling_window=52):
        """
        Remplace le StandardScaler (Data Leakage) par un Z-Score roulant.
        """
        cols_to_drop = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Date', 'Simple Returns', 'Log Returns', 'Log_Price',
                        'SMA_5', 'SMA_13', 'SMA_20', 'SMA_26', 'SMA_50', 'SMA_100', 'SMA_200', 'Regime']
        existing_drop = [c for c in cols_to_drop if c in train_data.columns]

        X_train = train_data.drop(columns=existing_drop)
        y_train = train_data['Target']
        X_test = test_data.drop(columns=existing_drop)
        y_test = test_data['Target']

        X_train_numeric = X_train.select_dtypes(include=[np.number])
        X_test_numeric = X_test.select_dtypes(include=[np.number])

        # --- CORRECTION DU BUG DE CONCATÉNATION ---
        # Si on passe le même dataset pour train et test (cas du backtest), on évite la duplication
        if X_train_numeric.index.equals(X_test_numeric.index):
            full_X = X_train_numeric.copy()
        else:
            # Concaténation des deux et suppression stricte des doublons d'index
            full_X = pd.concat([X_train_numeric, X_test_numeric])
            full_X = full_X[~full_X.index.duplicated(keep='first')]

        # Rolling Z-Score sur l'ensemble de la timeline pour une vraie modélisation out-of-sample
        rolling_mean = full_X.rolling(window=rolling_window, min_periods=1).mean()
        rolling_std = full_X.rolling(window=rolling_window, min_periods=1).std().replace(0, 1e-6)
        
        full_X_scaled = (full_X - rolling_mean) / rolling_std

        # On extrait proprement via l'index
        X_train_scaled = full_X_scaled.loc[X_train_numeric.index].values
        X_test_scaled = full_X_scaled.loc[X_test_numeric.index].values

        return X_train_scaled, y_train, X_test_scaled, y_test, None
    
    def reconstruct_prices(self, predictions, index):
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

    def reconstruct_hybrid_prices(self, arima_predictions, xgb_residual_predictions, index):
        total_predicted_log_returns = arima_predictions + xgb_residual_predictions
        return self.reconstruct_prices(total_predicted_log_returns, index)


class XGBoostModeler:
    """
    Architecture Ensemble pour XGBoost : Entraîne plusieurs modèles et moyenne 
    les résultats pour filtrer le bruit stochastique du marché.
    """
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, n_models=5):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        self.n_models = n_models
        self.models = []

    def train(self, X_train, y_train, use_custom_loss=False):
        self.models = []
        for i in range(self.n_models):
            # Injection de variance via le subsampling et des seeds différentes
            model = xgb.XGBRegressor(
                **self.params,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + i
            )
            model.fit(X_train, y_train)
            self.models.append(model)
        return self.models

    def predict(self, X_test):
        # Moyenne des prédictions de l'ensemble (Bagging effect)
        preds = np.array([model.predict(X_test) for model in self.models])
        return np.mean(preds, axis=0)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        correct_direction = np.sign(predictions) == np.sign(y_test)
        dir_accuracy = np.mean(correct_direction)
        
        return {
            'RMSE': rmse, 
            'MAE': mae, 
            'R2': r2, 
            'Directional_Accuracy': dir_accuracy,
            'Predictions': predictions
        }
    
    def get_feature_importance(self, feature_names):
        if not self.models:
            return None
        # Calcule l'importance moyenne à travers l'ensemble des arbres
        importances = np.mean([m.feature_importances_ for m in self.models], axis=0)
        
        if len(feature_names) != len(importances):
            return None
        
        df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        return df.sort_values(by='Importance', ascending=True)


class StatisticalModeler:
    def __init__(self, returns_series):
        self.returns = returns_series.dropna().sort_index()
        self.scale_factor = 100.0
        self.scaled_returns_series = self.returns * self.scale_factor
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
        try:
            if fixed_order is not None:
                self.arima_model = ARIMA(order=fixed_order, seasonal_order=None if not seasonal else (1, 0, 1, 12), suppress_warnings=True)
                self.arima_model.fit(self.train_values)
            else:
                self.arima_model = auto_arima(
                    self.train_values, start_p=1, start_q=1, max_p=5, max_q=5,
                    d=None, seasonal=seasonal, trace=False, error_action='ignore',  
                    suppress_warnings=True, stepwise=True
                )
            self.residuals = self.arima_model.resid()
            return self.arima_model
        except Exception as e:
            return None

    def fit_garch(self, p=1, q=1):
        if self.residuals is None: return None
        try:
            garch = arch_model(self.residuals, vol='Garch', p=p, q=q, mean='Zero', dist='Normal')
            self.garch_results = garch.fit(disp='off')
            return self.garch_results
        except Exception as e:
            return None

    def predict_next_step(self):
        if self.arima_model is None or self.garch_results is None:
            return 0.0, 0.0
        pred_scaled = self.arima_model.predict(n_periods=1)
        val_scaled = pred_scaled[0] if isinstance(pred_scaled, (list, np.ndarray)) else pred_scaled
        fcast = self.garch_results.forecast(horizon=1)
        if hasattr(fcast.variance, 'iloc'):
            var_scaled = fcast.variance.iloc[-1, 0]
        else:
            var_scaled = fcast.variance[-1, 0] if fcast.variance.ndim > 1 else fcast.variance[-1]
            
        vol_scaled = np.sqrt(var_scaled)
        return val_scaled / self.scale_factor, vol_scaled / self.scale_factor