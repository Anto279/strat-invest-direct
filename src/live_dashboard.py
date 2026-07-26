import pandas as pd
import numpy as np
import streamlit as st

from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models import ModelPreparator, StatisticalModeler, XGBoostModeler
from src.allocation import MarkowitzOptimizer
from src.strategy import Strategy

class LiveDashboard:
    def __init__(self, tickers, benchmark='^GSPC'):
        self.tickers = tickers
        self.benchmark = benchmark
        self.cost = 0.001 

    def get_latest_prices(self):
        loader = DataLoader(self.tickers, period="5d") 
        df = loader.load_data()
        last_prices = df['Close'].unstack(level='Ticker').iloc[-1]
        return last_prices

    def calculate_current_portfolio(self, positions_qty, cash, current_prices):
        total_value = cash
        asset_values = {}
        for ticker, qty in positions_qty.items():
            price = current_prices.get(ticker, 0.0)
            val = qty * price
            asset_values[ticker] = val
            total_value += val
            
        current_weights = {}
        if total_value > 0:
            for ticker, val in asset_values.items():
                current_weights[ticker] = val / total_value
        else:
            for ticker in self.tickers:
                current_weights[ticker] = 0.0
                
        return pd.Series(current_weights), total_value

    def run_live_analysis(self, current_weights, total_capital, lookback_years=3):
        status_log = []
        status_log.append("📡 Chargement des données de marché et macro-économiques...")
        
        loader = DataLoader(self.tickers, period="max")
        df_assets = loader.load_data()
        
        macro_tickers = {
            'VIX': '^VIX', 'Rates': '^TNX', 
            'DXY': 'DX-Y.NYB', 'OilVol': '^OVX'
        }
        loader_macro = DataLoader(list(macro_tickers.values()), period="max")
        df_macro = loader_macro.load_data()
        
        status_log.append("⚙️ Calcul des indicateurs techniques et transformation...")
        
        expected_returns = {}
        predicted_vols = {}
        
        # Horizon d'investissement aligné strictement sur le Walk-Forward Backtest
        HORIZON = 4 
        
        for ticker in self.tickers:
            try:
                df_single = df_assets.xs(ticker, level='Ticker', drop_level=False).copy()
            except KeyError:
                continue 
                
            fe = FeatureEngineer(df_single)
            if ticker in ['USO', 'GLD']:
                try: fe.add_context(df_macro.xs(macro_tickers['DXY'], level='Ticker'), prefix='DXY')
                except: pass
            if ticker == 'USO':
                try: fe.add_context(df_macro.xs(macro_tickers['OilVol'], level='Ticker'), prefix='OilVol')
                except: pass
            
            df_single = fe.add_moving_average(window=50, title='SMA_50')
            df_single = fe.add_moving_average(window=200, title='SMA_200')
            df_single = fe.add_rsi(window=14)
            df_single = fe.add_lags(lags=[1, 2, 3, 5])
            df_single = fe.add_bollinger_bands(window=20, num_std=2)   
            df_single = fe.add_macd()

            strat = Strategy(df_single)
            df_regime = strat.define_market_regime(col_ma_short='SMA_50', col_ma_long='SMA_200')
                
            prep = ModelPreparator(df_regime)
            prep.prepare_data()
            
            prep.data['Log Returns'] *= 100.0
            for c in prep.data.columns:
                if 'Ret_Lag' in c: prep.data[c] *= 100.0
                
            end_date = prep.data.index.get_level_values('Date').max()
            start_date = end_date - pd.DateOffset(years=lookback_years)
            
            mask = (prep.data.index.get_level_values('Date') >= start_date) & \
                   (prep.data.index.get_level_values('Date') <= end_date)
            
            train_df = prep.data.loc[mask].copy()
            
            if len(train_df) < 50:
                expected_returns[ticker] = 0.0
                predicted_vols[ticker] = 0.05
                continue

            # A. ARIMA (Tendance sur l'horizon)
            y_train = train_df['Log Returns'].dropna()
            stat_mod = StatisticalModeler(y_train / 100.0) 
            stat_mod.fit_arima()
            
            if stat_mod.arima_model:
                preds = stat_mod.arima_model.predict(n_periods=HORIZON)
                arima_pred = np.sum(preds) * 100.0 
            else:
                arima_pred = 0.0
                
            garch_res = stat_mod.fit_garch()
            if garch_res:
                forecasts = garch_res.forecast(horizon=HORIZON)
                vars_horizon = forecasts.variance.iloc[-1].values
                total_var = np.sum(vars_horizon)
                vol_pred = np.sqrt(total_var)
            else:
                vol_pred = y_train.std() * np.sqrt(HORIZON)

            # B. XGBoost (Correction des Résidus)
            residuals = stat_mod.residuals 
            min_len = min(len(y_train), len(residuals))
            s_resid = pd.Series(residuals[-min_len:], index=y_train.index[-min_len:])
            
            # Cible décalée alignée sur le backtest (Target cumulée)
            y_xgb_target = s_resid.rolling(window=HORIZON).sum().shift(-HORIZON).dropna()
            
            cols_drop = ['Log Returns', 'Target', 'SMA_50', 'SMA_200', 'Regime']
            common_idx = train_df.index.intersection(y_xgb_target.index)
            X_train = train_df.loc[common_idx].drop(columns=cols_drop, errors='ignore')
            y_train_xgb = y_xgb_target.loc[common_idx]
            
            if len(X_train) > 20:
                xgb_mod = XGBoostModeler(n_estimators=20, max_depth=2, n_models=5)
                xgb_mod.train(X_train, y_train_xgb, use_custom_loss=False)
                
                X_last = train_df.iloc[[-1]].drop(columns=cols_drop, errors='ignore')
                xgb_pred = xgb_mod.predict(X_last)[0]
            else:
                xgb_pred = 0.0
                
            final_ret_pct = arima_pred + xgb_pred
            
            expected_returns[ticker] = final_ret_pct / 100.0
            predicted_vols[ticker] = vol_pred / 100.0
            
        status_log.append("✅ Modèles entraînés (ARIMA + XGBoost).")
        status_log.append("⚖️ Calcul de l'allocation optimale (Markowitz)...")
        
        recent_data = df_assets.loc[df_assets.index.get_level_values('Date') >= (end_date - pd.DateOffset(years=1))]
        pivoted = recent_data.pivot_table(index='Date', columns='Ticker', values='Close').pct_change().dropna()
        corr_matrix = pivoted.corr()
        
        vols = pd.Series(predicted_vols).reindex(self.tickers).fillna(0.01)
        D = np.diag(vols.values)
        cov_matrix = pd.DataFrame(D @ corr_matrix.values @ D, index=self.tickers, columns=self.tickers)
        
        mu = pd.Series(expected_returns).reindex(self.tickers).fillna(0.0)
        bounds = strat.get_current_bounds(self.tickers) 
        
        opt = MarkowitzOptimizer()
        target_weights = opt.optimize_portfolio(
            expected_returns=mu,
            cov_matrix=cov_matrix,
            current_weights=current_weights.values, 
            constraints_bounds=bounds,
            transaction_cost=self.cost
        )
        
        df_alloc = pd.DataFrame({
            'Actif': target_weights.index,
            'Allocation Actuelle (%)': current_weights.reindex(target_weights.index).fillna(0.0).values * 100,
            'Allocation Cible (%)': target_weights.values * 100,
            'Rendement Espéré (Horizon)': mu.values,
            'Volatilité (Horizon)': vols.values * 100
        }).set_index('Actif')
        
        df_alloc['Différence (%)'] = df_alloc['Allocation Cible (%)'] - df_alloc['Allocation Actuelle (%)']
        df_alloc['Ordre ($)'] = (df_alloc['Différence (%)'] / 100) * total_capital
        
        current_prices = self.get_latest_prices()
        df_alloc['Prix Actuel'] = df_alloc.index.map(current_prices)
        df_alloc['Ordre (Qté)'] = df_alloc['Ordre ($)'] / df_alloc['Prix Actuel']

        regimes_dict = dict(zip(self.tickers, bounds))
        
        return df_alloc, status_log, regimes_dict