import pandas as pd
import numpy as np
import streamlit as st

# Importation des modules du projet (Architecture identique au Backtest)
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models import ModelPreparator, StatisticalModeler, XGBoostModeler
from src.allocation import MarkowitzOptimizer
from src.strategy import Strategy

class LiveDashboard:
    def __init__(self, tickers, benchmark='^GSPC'):
        self.tickers = tickers
        self.benchmark = benchmark
        self.cost = 0.001  # Frais standard

    def get_latest_prices(self):
        """Récupère simplement le dernier prix de clôture pour chaque actif."""
        loader = DataLoader(self.tickers, period="5d") # On prend juste assez pour avoir le dernier close
        df = loader.load_data()
        # On prend la dernière ligne disponible
        last_prices = df['Close'].unstack(level='Ticker').iloc[-1]
        return last_prices

    def calculate_current_portfolio(self, positions_qty, cash, current_prices):
        """
        Convertit les quantités (nombre d'actions) en Poids (%) pour l'optimiseur.
        """
        total_value = cash
        asset_values = {}
        
        for ticker, qty in positions_qty.items():
            price = current_prices.get(ticker, 0.0)
            val = qty * price
            asset_values[ticker] = val
            total_value += val
            
        # Calcul des poids actuels
        current_weights = {}
        if total_value > 0:
            for ticker, val in asset_values.items():
                current_weights[ticker] = val / total_value
        else:
            # Si portefeuille vide ou net 0, poids nuls
            for ticker in self.tickers:
                current_weights[ticker] = 0.0
                
        return pd.Series(current_weights), total_value

    def run_live_analysis(self, current_weights, total_capital, lookback_years=3):
        """
        Exécute le pipeline COMPLET (Data -> Features -> Hybrid Model -> Optimisation)
        Exactement comme dans le Backtest Walk-Forward.
        """
        status_log = [] # Pour afficher la progression dans l'UI
        
        # 1. CHARGEMENT DES DONNÉES (FULL HISTORY pour Feature Engineering correct)
        # ==============================================================================
        status_log.append("📡 Chargement des données de marché et macro-économiques...")
        
        # Actifs
        loader = DataLoader(self.tickers, period="max")
        df_assets = loader.load_data()
        
        # Macro (Indispensable pour XGBoost comme dans le Backtest)
        macro_tickers = {
            'VIX': '^VIX', 'Rates': '^TNX', 
            'DXY': 'DX-Y.NYB', 'OilVol': '^OVX'
        }
        loader_macro = DataLoader(list(macro_tickers.values()), period="max")
        df_macro = loader_macro.load_data()
        
        # 2. FEATURE ENGINEERING & PRÉPARATION
        # ==============================================================================
        status_log.append("⚙️ Calcul des indicateurs techniques et transformation...")
        
        expected_returns = {}
        predicted_vols = {}
        
        # On boucle sur chaque actif pour préparer ses données spécifiquement
        for ticker in self.tickers:
            # Isolation de l'actif
            try:
                df_single = df_assets.xs(ticker, level='Ticker', drop_level=False).copy()
            except KeyError:
                continue # Skip si ticker pas trouvé
                
            # Features
            fe = FeatureEngineer(df_single)
            
            # Ajout Context Macro (Comme dans Backtest)
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
            # 2. Bollinger (Volatilité - Crucial pour Pétrole/Commodities)
            df_single = fe.add_bollinger_bands(window=20, num_std=2)   
            # 3. MACD (Momentum - Crucial pour CAC40/Indices)
            df_single = fe.add_macd()

            strat = Strategy(df_single)
            df_regime = strat.define_market_regime(col_ma_short='SMA_50', col_ma_long='SMA_200')
                
            # Préparation (Log Returns, Cleaning)
            prep = ModelPreparator(df_regime)
            prep.prepare_data()
            
            # --- SCALING x100 (CRITIQUE : MÊME ÉCHELLE QUE BACKTEST) ---
            prep.data['Log Returns'] *= 100.0
            # Scale des lags
            for c in prep.data.columns:
                if 'Ret_Lag' in c: prep.data[c] *= 100.0
                
            # Définition de la fenêtre d'entraînement (Lookback)
            # On s'arrête à la date d'aujourd'hui (dernière date dispo)
            end_date = prep.data.index.get_level_values('Date').max()
            start_date = end_date - pd.DateOffset(years=lookback_years)
            
            # Masque d'entraînement
            mask = (prep.data.index.get_level_values('Date') >= start_date) & \
                   (prep.data.index.get_level_values('Date') <= end_date)
            
            train_df = prep.data.loc[mask].copy()
            
            if len(train_df) < 50:
                expected_returns[ticker] = 0.0
                predicted_vols[ticker] = 0.05
                continue

            # 3. MODÉLISATION HYBRIDE (ARIMA + XGBoost)
            # ==============================================================================
            
            # A. ARIMA (Tendance)
            y_train = train_df['Log Returns'].dropna()
            # On divise par 100 pour StatisticalModeler qui attend des décimales pour fit_garch
            # Mais StatisticalModeler refait x100 en interne pour ARIMA... 
            # Pour être cohérent avec votre classe : on passe y/100
            stat_mod = StatisticalModeler(y_train / 100.0) 
            
            stat_mod.fit_arima()
            
            # Prédiction T+1 (Semaine prochaine)
            if stat_mod.arima_model:
                # predict renvoie en % car la classe le gère
                arima_pred = stat_mod.arima_model.predict(n_periods=1)[0] * 100.0 
            else:
                arima_pred = 0.0
                
            # Volatilité (GARCH)
            garch_res = stat_mod.fit_garch()
            if garch_res:
                # Forecast variance à 1 pas
                vol_pred = np.sqrt(garch_res.forecast(horizon=1).variance.iloc[-1].values[0])
            else:
                vol_pred = y_train.std() # Fallback

            # B. XGBoost (Correction des Résidus)
            # 1. Calcul des résidus sur le train (Target)
            residuals = stat_mod.residuals # Déjà aligné avec y_train normalement
            # Alignement sécurité
            min_len = min(len(y_train), len(residuals))
            s_resid = pd.Series(residuals[-min_len:], index=y_train.index[-min_len:])
            
            # 2. Création Target Décalée (On veut prédire le résidu de t+1 avec les features de t)
            y_xgb_target = s_resid.shift(-1).dropna()
            
            # 3. Features
            # On doit dropper les colonnes non-features
            cols_drop = ['Log Returns', 'Target', 'SMA_50', 'SMA_200', 'Regime']
            # On s'assure d'avoir les features alignées
            common_idx = train_df.index.intersection(y_xgb_target.index)
            X_train = train_df.loc[common_idx].drop(columns=cols_drop, errors='ignore')
            y_train_xgb = y_xgb_target.loc[common_idx]
            
            if len(X_train) > 20:
                xgb_mod = XGBoostModeler(n_estimators=20, max_depth=2) # Paramètres standards
                xgb_mod.train(X_train, y_train_xgb, use_custom_loss=False)
                
                # 4. Prédiction T+1
                # On prend la DERNIÈRE ligne de features disponible (Celle d'aujourd'hui)
                # Pour prédire la correction de DEMAIN
                X_last = train_df.iloc[[-1]].drop(columns=cols_drop, errors='ignore')
                xgb_pred = xgb_mod.predict(X_last)[0]
            else:
                xgb_pred = 0.0
                
            # 4. COMBINAISON FINALE
            # ==============================================================================
            # Retour final = ARIMA (Tendance) + XGBoost (Correction)
            # Tout est en pourcentage ici (ex: 0.5 pour 0.5%)
            final_ret_pct = arima_pred + xgb_pred
            
            # Stockage (On repasse en décimal pour l'optimiseur Markowitz)
            expected_returns[ticker] = final_ret_pct / 100.0
            predicted_vols[ticker] = vol_pred / 100.0
            
        status_log.append("✅ Modèles entraînés (ARIMA + XGBoost).")

        # 4. OPTIMISATION DE PORTEFEUILLE
        # ==============================================================================
        status_log.append("⚖️ Calcul de l'allocation optimale (Markowitz)...")
        
        # Matrice de Covariance (Estimée via Corrélation Historique + Volatilité Prédite GARCH)
        # On prend les 1 dernières années pour la corrélation
        recent_data = df_assets.loc[df_assets.index.get_level_values('Date') >= (end_date - pd.DateOffset(years=1))]
        pivoted = recent_data.pivot_table(index='Date', columns='Ticker', values='Close').pct_change().dropna()
        corr_matrix = pivoted.corr()
        
        # Reconstruction Matrice Covariance : D * Corr * D
        vols = pd.Series(predicted_vols).reindex(self.tickers).fillna(0.01)
        D = np.diag(vols.values)
        cov_matrix = pd.DataFrame(D @ corr_matrix.values @ D, index=self.tickers, columns=self.tickers)
        
        # Vecteur Rendements Espérés
        mu = pd.Series(expected_returns).reindex(self.tickers).fillna(0.0)
        
        # Filtre de Régime (Safety First)
        # On utilise la classe Strategy pour obtenir les bornes (Long/Short/Neutral)
        # Il faut passer tout l'historique à Strategy pour qu'elle calcule les MM50/200 correctement
        bounds = strat.get_current_bounds(self.tickers) # Renvoie (min, max) pour chaque actif
        
        # Optimiseur
        opt = MarkowitzOptimizer()
        target_weights = opt.optimize_portfolio(
            expected_returns=mu,
            cov_matrix=cov_matrix,
            current_weights=current_weights.values, # Pour limiter le turnover si besoin
            constraints_bounds=bounds,
            transaction_cost=self.cost
        )
        
        # 5. GÉNÉRATION DES ORDRES
        # ==============================================================================
        df_alloc = pd.DataFrame({
            'Actif': target_weights.index,
            'Allocation Actuelle (%)': current_weights.reindex(target_weights.index).fillna(0.0).values * 100,
            'Allocation Cible (%)': target_weights.values * 100,
            'Rendement Espéré (Semaine)': mu.values,
            'Volatilité (Semaine)': vols.values * 100
        }).set_index('Actif')
        
        # Calcul des ordres en $ et Quantité
        df_alloc['Différence (%)'] = df_alloc['Allocation Cible (%)'] - df_alloc['Allocation Actuelle (%)']
        df_alloc['Ordre ($)'] = (df_alloc['Différence (%)'] / 100) * total_capital
        
        # On récupère les prix actuels pour estimer la quantité à bouger
        current_prices = self.get_latest_prices()
        df_alloc['Prix Actuel'] = df_alloc.index.map(current_prices)
        df_alloc['Ordre (Qté)'] = df_alloc['Ordre ($)'] / df_alloc['Prix Actuel']

        regimes_dict = dict(zip(self.tickers, bounds))
        
        return df_alloc, status_log, regimes_dict