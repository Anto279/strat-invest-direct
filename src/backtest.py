import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.features import FeatureEngineer
from src.models import ModelPreparator, StatisticalModeler, XGBoostModeler
from src.allocation import MarkowitzOptimizer
from src.strategy import Strategy, HMMStrategy
from src.risk_metrics import calculate_gdr_metrics
from src.data_loader import DataLoader

class WalkForwardBacktest:
    """
    Moteur de Backtest Walk-Forward.
    Stratégie : "Lazy Trading" via Buffer.
    """
    
    def __init__(self, full_data, tickers, benchmark_ticker='^GSPC', initial_capital=10000, transaction_cost=0.001):
        self.data = full_data.sort_index(level='Date')
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.capital = initial_capital
        self.cost = transaction_cost # Vrai coût payé (ex: 0.10%)
        
        self.history = []
        self.weights_history = []

    def run(self, start_date, step_weeks=4, turnover_buffer=0.40, regime_method='sma'):
        available_dates = self.data.index.get_level_values('Date').unique().sort_values()
        
        try:
            start_idx = available_dates.get_loc(pd.Timestamp(start_date))
        except KeyError:
            start_idx = available_dates.searchsorted(pd.Timestamp(start_date))
        
        print(f"--- Backtest Démarré ({len(available_dates) - start_idx} périodes) ---")
        
        current_capital = self.capital
        current_weights = pd.Series(0.0, index=self.tickers)

        # --- PARAMÈTRE TAUX SANS RISQUE FIXE ---
        RISK_FREE_RATE_ANNUAL = 0.02  # 2% par an (Ajustez à 0.04 ou 0.05 pour être réaliste aujourd'hui)
        
        for i in range(start_idx, len(available_dates) - 1, step_weeks):
            current_date = available_dates[i]
            next_idx = min(i + step_weeks, len(available_dates) - 1)
            next_date = available_dates[next_idx]
            
            # --- 1. ALLOCATION ---
            window_start = available_dates[max(0, i - 156)] # 3 ans de données pour l'estimation
            mask_history = (
                (self.data.index.get_level_values('Date') >= window_start) & 
                (self.data.index.get_level_values('Date') <= current_date)
            )
            history_df = self.data[mask_history].copy()
            
            raw_target_weights = self._generate_allocation(history_df, current_weights, horizon=4, regime_method=regime_method)
            
            # --- 2. BUFFER ---
            proposed_turnover = np.sum(np.abs(raw_target_weights - current_weights))
            
            if proposed_turnover < turnover_buffer:
                target_weights = current_weights
                actual_turnover = 0.0
            else:
                target_weights = raw_target_weights
                actual_turnover = proposed_turnover
            
            # --- 3. EXÉCUTION ---
            mask_period = (
                (self.data.index.get_level_values('Date') > current_date) & 
                (self.data.index.get_level_values('Date') <= next_date)
            )
            if not mask_period.any(): break
            period_data = self.data[mask_period]
            
            asset_period_returns = period_data.groupby('Ticker')['Log Returns'].sum()
            asset_period_returns = asset_period_returns.reindex(self.tickers).fillna(0)

            # Poids du cash (Ce qui n'est pas investi)
            total_invested_weight = np.sum(np.abs(target_weights))
            cash_weight = max(0.0, 1.0 - total_invested_weight)

            # --- CORRECTION ICI : Rémunération du Cash ---
            
            # 1. Calcul de la durée réelle de la période (pour gérer la dernière semaine si incomplète)
            actual_weeks = next_idx - i
            
            # 2. Rendement Simple du Cash sur la période : (1 + r_annuel)^(semaines/52) - 1
            risk_free_period_ret = (1 + RISK_FREE_RATE_ANNUAL) ** (actual_weeks / 52.0) - 1
            
            # 3. Conversion en Log Return pour additionner avec les actifs
            risk_free_log_ret = np.log(1 + risk_free_period_ret)
            
            # 4. Rendement Portefeuille GLOBAL
            # = (Poids Actifs * Rendement Actifs) + (Poids Cash * Rendement Cash)
            invested_return = np.dot(target_weights.values, asset_period_returns.values)
            cash_return = cash_weight * risk_free_log_ret
            
            port_log_ret = invested_return + cash_return
            
            # Coûts de transaction
            cost_amount = actual_turnover * self.cost
            
            capital_after_cost = current_capital * (1 - cost_amount)
            new_capital = capital_after_cost * np.exp(port_log_ret)
            
            net_return = (new_capital / current_capital) - 1
            
            try: bench_ret = period_data.xs(self.benchmark_ticker, level='Ticker')['Log Returns'].sum()
            except: bench_ret = 0.0
            
            self.history.append({
                'Date': next_date,
                'Capital': new_capital,
                'Net_Return': net_return,
                'Turnover': actual_turnover,
                'Transaction_Costs': cost_amount * current_capital,
                'Benchmark_Log_Return': bench_ret
            })
            
            self.weights_history.append(target_weights.rename(current_date))
            
            # --- 4. GESTION DU DRIFT ---
            drift_factors = np.exp(asset_period_returns)
            portfolio_growth_factor = np.exp(port_log_ret) # Inclut maintenant la croissance du cash
            
            # Les poids des actifs diminuent légèrement si le portefeuille grossit grâce au cash (effet de dilution normal)
            drifted_weights = (target_weights * drift_factors) / portfolio_growth_factor
            
            current_weights = drifted_weights
            current_capital = new_capital

        return pd.DataFrame(self.history).set_index('Date')

    def _generate_allocation(self, history_df, current_weights, horizon=4, regime_method='sma'):
        expected_returns = {}
        predicted_vols = {}

        # Dictionnaire pour stocker la proba Bear de CHAQUE actif
        assets_bear_probs = {}
        
        for ticker in self.tickers:
            df_asset = history_df.xs(ticker, level='Ticker', drop_level=False).copy()

            # --- CALCUL DU RÉGIME HMM SPÉCIFIQUE À L'ACTIF ---
            # On le fait ici, dans la boucle, pour chaque actif indépendamment
            if regime_method == 'hmm':
                try:
                    # On s'assure qu'il y a assez de données (ex: 1 an)
                    if len(df_asset) > 52:
                        hmm = HMMStrategy(df_asset)
                        # Entraînement sur l'historique de CET actif
                        df_hmm = hmm.fit_predict(vol_window=12)
                        # On prend la dernière proba disponible
                        p_bear = df_hmm['Bear_Prob'].iloc[-1]
                        assets_bear_probs[ticker] = p_bear
                    else:
                        assets_bear_probs[ticker] = 0.0 # Pas assez de data = On suppose Bull/Neutre
                except Exception as e:
                    # Fallback en cas d'erreur de convergence HMM
                    assets_bear_probs[ticker] = 0.0

            prep = ModelPreparator(df_asset)
            cols_scale = ['Log Returns', 'Target'] + [c for c in df_asset.columns if 'Ret_Lag' in c]
            for c in cols_scale:
                if c in df_asset.columns: df_asset[c] *= 100.0
            
            train_df = df_asset
            y_train = train_df['Log Returns'].dropna()
            
            if len(y_train) < 30: 
                expected_returns[ticker] = 0.0; predicted_vols[ticker] = 0.01; continue

            stat_mod = StatisticalModeler(y_train / 100.0)
            stat_mod.fit_arima()
            garch_res = stat_mod.fit_garch(p=1, q=1)
            
            # --- 1. PRÉDICTION ARIMA (Cumul Horizon) ---
            if stat_mod.arima_model:
                # On prédit 'horizon' semaines (ex: 4)
                preds = stat_mod.arima_model.predict(n_periods=horizon)
                # La performance sur 4 semaines est la somme des log-returns hebdo prédits
                val = np.sum(preds)
                arima_pred = val / 100.0
            else: arima_pred = 0.0
            
            # --- 2. PRÉDICTION GARCH (Volatilité Horizon) ---
            if garch_res:
                # On projette la variance sur l'horizon
                forecasts = garch_res.forecast(horizon=horizon)
                # On récupère les variances prédites pour t+1, t+2... t+horizon
                # .iloc[-1] donne la prévision faite à la dernière date dispo
                vars_horizon = forecasts.variance.iloc[-1].values
                # Variance totale = Somme des variances (si indépendance temporelle des chocs)
                total_var = np.sum(vars_horizon)
                garch_vol = np.sqrt(total_var) / 100.0
            else: 
                # Fallback : Racine carrée du temps (Square Root of Time Rule)
                garch_vol = 0.01 * np.sqrt(horizon)

            # --- 3. XGBoost (Target Cumulée) ---
            resids = stat_mod.residuals
            resid_index = y_train.index[-len(resids):]
            s_resids = pd.Series(resids, index=resid_index)
            
            # TARGET MODIFIÉE : Somme cumulée des résidus des 'horizon' prochaines semaines
            # rolling(4).sum() fait la somme de (t-3, t-2, t-1, t). 
            # Pour avoir la somme de (t+1, t+2, t+3, t+4), on décale en arrière de 'horizon'.
            y_target_resid = s_resids.rolling(window=horizon).sum().shift(-horizon).dropna()
            # y_target_resid = s_resids.shift(-1).dropna()
            
            X_full_np, _, _, _, _ = prep.get_X_y(train_df, train_df)
            X_full_df = pd.DataFrame(X_full_np, index=train_df.index)
            
            # Alignement
            common_index = X_full_df.index.intersection(y_target_resid.index)
            
            window_xgb = 100 # 3 ans
            if len(common_index) > window_xgb:
                common_index = common_index[-window_xgb:]
            
            if len(common_index) < 10:
                xgb_pred = 0.0
            else:
                X_train_aligned = X_full_df.loc[common_index]
                y_train_aligned = y_target_resid.loc[common_index]
                
                xgb_mod = XGBoostModeler(n_estimators=20, max_depth=2)
                xgb_mod.train(X_train_aligned, y_train_aligned, use_custom_loss=False)
                
                # Prédiction
                X_last = X_full_df.iloc[[-1]]
                xgb_pred = xgb_mod.predict(X_last)[0]
            
            # Rendement Total Espéré sur 4 semaines
            expected_returns[ticker] = arima_pred + (xgb_pred / 100.0)
            predicted_vols[ticker] = garch_vol

        # Matrice de Covariance adaptée
        # On garde la corrélation hebdomadaire (souvent stable) mais on applique la Volatilité Mensuelle
        recent = history_df.pivot_table(index='Date', columns='Ticker', values='Log Returns').tail(52)
        recent = recent.reindex(columns=self.tickers).fillna(0)
        corr = recent.corr().fillna(0)
        
        # D contient maintenant les volatilités sur 4 semaines
        D = np.diag([predicted_vols.get(t, 0.01) for t in self.tickers])
        cov_matrix = pd.DataFrame(D @ corr.values @ D, index=self.tickers, columns=self.tickers)
        
        # === CONTRAINTES & BORNES ===
        bounds = None
        
        if regime_method == 'sma':
            # SMA : On calcule les bornes strictes (ex: 0,0 si baissier)
            # Strategy.get_current_bounds fait déjà le travail actif par actif
            strat = Strategy(history_df)
            strat.define_market_regime('SMA_50', 'SMA_200')
            bounds = strat.get_current_bounds(self.tickers)
            # En SMA, on n'utilise pas bear_probs dans l'optimiseur
            assets_bear_probs = None 
            
        elif regime_method == 'hmm':
            # Méthode HMM : Bornes calculées via les probabilités accumulées
            # On utilise la méthode statique qu'on vient de créer
            final_bounds = HMMStrategy.get_bounds_from_probabilities(
                self.tickers, 
                assets_bear_probs, 
                threshold=0.90 # Seuil strict
            )

        # 3. OPTIMISATION
        opt = MarkowitzOptimizer()
        mu = pd.Series(expected_returns).reindex(self.tickers).fillna(0)
        
        # Notez bien : bear_probs=None
        # Car le risque est maintenant géré par les bornes strictes (final_bounds)
        # On ne veut pas scaler les poids une deuxième fois.
        
        weights = opt.optimize_portfolio(
            expected_returns=mu,
            cov_matrix=cov_matrix,
            current_weights=current_weights.values,
            constraints_bounds=final_bounds, # Bornes HMM ou SMA
            transaction_cost=0.0001,
            regime_method='sma' , # On utilise les bornes, pas le scaling 'sma'
            bear_probs=None 
        )
        
        return weights

    def run_with_tranching(self, start_date, step_weeks=4, turnover_buffer=0.40, regime_method='sma'):
        """
        Exécute 4 backtests décalés et retourne les résultats bruts.
        Ajoute le point de départ (T=0) pour que les graphiques commencent bien au capital initial.
        """
        print(f"--- Démarrage du Tranching Séparé ({step_weeks} tranches) ---")
        print(f"--- Démarrage du Tranching Séparé | Mode: {regime_method} ---")
        
        all_tranches = {}
        start_dt = pd.to_datetime(start_date)
        
        for i in range(step_weeks):
            offset_date = start_dt + pd.Timedelta(weeks=i)
            
            # 1. Reset
            self.history = []
            self.weights_history = []
            
            # 2. Run
            res = self.run(
                start_date=offset_date.strftime('%Y-%m-%d'),
                step_weeks=step_weeks, 
                turnover_buffer=turnover_buffer,
                regime_method=regime_method
            )
            
            # --- CORRECTION : AJOUT DU POINT INITIAL (T=0) ---
            # On crée une ligne pour la date de départ exacte avec le capital initial
            start_row = pd.DataFrame({
                'Capital': [self.capital],
                'Net_Return': [0.0],
                'Turnover': [0.0],
                'Transaction_Costs': [0.0],
                'Benchmark_Log_Return': [0.0]
            }, index=[offset_date]) # Index = Date de départ de la tranche
            
            # On colle au début du résultat
            res_full = pd.concat([start_row, res])
            
            # 3. Stockage
            tranche_data = {
                'df': res_full,
                'weights': list(self.weights_history)
            }
            all_tranches[f'Tranche_{i}'] = tranche_data
            
        return all_tranches

    def get_metrics(self, df_results):
        if 'Strategy_Equity' not in df_results.columns:
            df_results['Strategy_Equity'] = self.capital * (1 + df_results['Net_Return']).cumprod()
            
        # 1. Performance Stratégie
        total_ret = (df_results['Strategy_Equity'].iloc[-1] / self.capital) - 1
        days = (df_results.index[-1] - df_results.index[0]).days
        years = days / 365.25
        if years > 0: 
            cagr = (1 + total_ret) ** (1 / years) - 1
        else: 
            cagr = 0
        
        # 2. Volatilité & Sharpe
        freq_factor = len(df_results) / years if years > 0 else 12
        vol = df_results['Net_Return'].std() * np.sqrt(freq_factor)
        sharpe = (cagr - 0.02) / vol if vol > 0 else 0
        
        # 3. Drawdown & Win Rate
        cum = df_results['Strategy_Equity']
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        win_rate = len(df_results[df_results['Net_Return'] > 0]) / len(df_results)

        # 4. ALPHA & BETA (Nouveau)
        # On convertit les log returns du benchmark en rendements simples pour la comparaison
        bench_simple_ret = np.exp(df_results['Benchmark_Log_Return']) - 1
        strat_simple_ret = df_results['Net_Return']

        # Matrice de Covariance [0,1] = covar, [1,1] = var(bench)
        cov_matrix = np.cov(strat_simple_ret, bench_simple_ret)
        
        if cov_matrix[1, 1] > 0:
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        else:
            beta = 0
            
        # Calcul du CAGR Benchmark pour déduire l'Alpha
        total_ret_bench = (1 + bench_simple_ret).prod() - 1
        if years > 0:
            cagr_bench = (1 + total_ret_bench) ** (1 / years) - 1
        else:
            cagr_bench = 0
            
        # Alpha de Jensen (Annualisé) : R_strat - (R_risk_free + Beta * (R_bench - R_risk_free))
        # On suppose Risk Free ~ 2% pour être conservateur, ou 0% pour simplifier. Prenons 0 par défaut ici.
        alpha = cagr - (0.02 + beta * cagr_bench)

        # Récupération des rendements (Net_Return est souvent mensuel ou hebdo dans votre code)
        rets = df_results['Net_Return']
        
        # APPEL DES MESURES DE RISQUE GBR
        risk_metrics = calculate_gdr_metrics(rets, confidence_level=0.95)

        return {
            "Total Return": total_ret,
            "CAGR (Annuel)": cagr,
            "Volatilité": vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Win Rate": win_rate,
            "Beta": beta,
            "Alpha": alpha,
            "VaR (95%)": risk_metrics["VaR Historique (95%)"],
            "CVaR (95%)": risk_metrics["CVaR / Expected Shortfall (95%)"],
            "Skewness": risk_metrics["Skewness"], # Utile pour savoir si les gains sont asymétriques
            "Kurtosis": risk_metrics["Kurtosis"]  # La Kurtosis mesure si les événements extrêmes (très rares) arrivent plus souvent que prévu par la loi Normale
        }
    
    def plot_results(self, df_results):
        """ 
        Affiche la performance comparée avec :
        1. S&P 500 (100% Actions)
        2. Taux Sans Risque (Cash/BIL)
        3. Portefeuille 60/40 (Actions/Bonds)
        """
        import yfinance as yf # Import local nécessaire pour télécharger les benchmarks
        
        # 1. Calcul Stratégie
        if 'Strategy_Equity' not in df_results.columns:
            df_results['Strategy_Equity'] = self.capital * (1 + df_results['Net_Return']).cumprod()
            
        # 2. Calcul Benchmark 1 : S&P 500 (Déjà présent)
        if 'Benchmark_Equity' not in df_results.columns:
             b_cum = df_results['Benchmark_Log_Return'].cumsum()
             df_results['Benchmark_Equity'] = self.capital * np.exp(b_cum)
             
        # --- CALCUL DES NOUVEAUX BENCHMARKS ---
        try:
            # On définit une plage de date large pour éviter les erreurs
            start_d = df_results.index[0] - pd.Timedelta(days=7)
            end_d = df_results.index[-1] + pd.Timedelta(days=7)
            
            # On télécharge BIL (Cash) et TLT (Bonds pour le 60/40)
            tickers_bench = ['BIL', 'TLT']
            bench_data = yf.download(tickers_bench, start=start_d, end=end_d, progress=False)['Close']
            
            # On réaligne sur les dates exactes du backtest (ffill pour combler les trous)
            bench_aligned = bench_data.reindex(df_results.index, method='ffill')
            
            # A. BENCHMARK 2 : Taux Sans Risque (BIL)
            # Normalisation base 100 sur le capital initial
            risk_free_curve = (bench_aligned['BIL'] / bench_aligned['BIL'].iloc[0]) * self.capital
            
            # B. BENCHMARK 3 : 60/40 (Buy & Hold)
            # 60% de Capital sur le S&P500 (déjà calculé dans Benchmark_Equity)
            # 40% de Capital sur le TLT (Bonds)
            # Note : Benchmark_Equity représente déjà la courbe du S&P normalisée sur self.capital
            equity_part = df_results['Benchmark_Equity'] * 0.60
            bond_part = (bench_aligned['TLT'] / bench_aligned['TLT'].iloc[0]) * self.capital * 0.40
            
            balanced_60_40 = equity_part + bond_part
            
        except Exception as e:
            # Si échec téléchargement (hors ligne), on crée des lignes plates pour ne pas planter
            print(f"Warning - Benchmark download failed: {e}")
            risk_free_curve = pd.Series(self.capital, index=df_results.index)
            balanced_60_40 = pd.Series(self.capital, index=df_results.index)

        # --- GRAPHIQUE ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=("Comparaison de Performance (Capital)", "Drawdown"))
        
        # 1. TA STRATÉGIE (Vert, ligne solide)
        fig.add_trace(go.Scatter(
            x=df_results.index, y=df_results['Strategy_Equity'], 
            name='Ma Stratégie', 
            line=dict(color='green', width=2)
        ), row=1, col=1)
        
        # 2. Bench 1 : S&P 500 (Gris pointillé)
        fig.add_trace(go.Scatter(
            x=df_results.index, y=df_results['Benchmark_Equity'], 
            name='S&P 500', # S&P 500 (100% Actions)
            line=dict(color='grey', dash='dot', width=1)
        ), row=1, col=1)

        # 3. Bench 2 : Cash / Risk Free (Orange tireté)
        fig.add_trace(go.Scatter(
            x=df_results.index, y=risk_free_curve, 
            name='Cash (Risk Free)', 
            line=dict(color='orange', dash='dash', width=1.5)
        ), row=1, col=1)

        # 4. Bench 3 : 60/40 (Bleu tireté)
        # fig.add_trace(go.Scatter(
        #    x=df_results.index, y=balanced_60_40, 
        #    name='Portfolio 60/40 (Actions/Bonds)', 
        #    line=dict(color='#3366CC', dash='dashdot', width=1.5)
        #), row=1, col=1)
        
        # --- DRAWDOWN ---
        dd = (df_results['Strategy_Equity'] - df_results['Strategy_Equity'].cummax()) / df_results['Strategy_Equity'].cummax()
        fig.add_trace(go.Scatter(
            x=df_results.index, y=dd, 
            name='Drawdown Stratégie', 
            fill='tozeroy', 
            line=dict(color='red', width=1)
        ), row=2, col=1)
        
        fig.update_layout(height=650, title="Backtest Walk-Forward & Benchmarks", hovermode="x unified")
        
        # Format axes
        fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", tickformat=".0%", row=2, col=1)
        
        return fig
    
    def plot_cost_analysis(self, df_results):
        """ Affiche l'analyse des coûts de transaction """
        cum_costs = df_results['Transaction_Costs'].cumsum()
        
        # Pourcentage des frais par rapport au capital courant
        # Si on a stocké Cost_Percent dans run(), on l'utilise, sinon on recalcule
        if 'Cost_Percent' in df_results.columns:
            pct_costs = df_results['Cost_Percent'] * 100
        else:
            # Fallback approx
            pct_costs = (df_results['Transaction_Costs'] / df_results['Capital']) * 100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            subplot_titles=("Frais Cumulés ($)", "Impact Périodique des Frais (% du Capital)"),
                            vertical_spacing=0.1)

        # 1. Frais Cumulés (Area)
        fig.add_trace(go.Scatter(
            x=df_results.index, y=cum_costs, 
            fill='tozeroy', name='Frais Cumulés',
            line=dict(color='orange')
        ), row=1, col=1)

        # 2. Frais Périodiques (Bar)
        fig.add_trace(go.Bar(
            x=df_results.index, y=pct_costs,
            name='% Frais (Période)',
            marker_color='salmon'
        ), row=2, col=1)

        fig.update_layout(height=500, title="Analyse de l'Impact des Frais", hovermode="x unified")
        fig.update_yaxes(title_text="$", row=1, col=1)
        fig.update_yaxes(title_text="%", row=2, col=1)
        
        return fig

    def plot_allocation_evolution(self):
        """ Affiche l'évolution des poids avec la poche Cash (Calculée sur l'exposition brute) """
        if not self.weights_history:
            return None
            
        # 1. Conversion
        df_weights = pd.DataFrame(self.weights_history)
        df_weights.index = pd.to_datetime(df_weights.index)
        
        # 2. CALCUL CORRECT DU CASH (Basé sur l'Exposition Brute)
        # On calcule la somme des valeurs ABSOLUES des poids.
        # Ex: Si j'ai -0.5 (Short) et +0.2 (Long), mon exposition brute est 0.7.
        # Il me reste donc 0.3 (30%) de Cash libre.
        current_exposure = df_weights.abs().sum(axis=1)
        
        df_weights['Cash'] = 1.0 - current_exposure
        
        # Nettoyage (Si exposition > 100% à cause d'un arrondi, Cash = 0)
        df_weights['Cash'] = df_weights['Cash'].apply(lambda x: x if x > 1e-4 else 0)

        fig = go.Figure()
        
        # 4. Affichage des Actifs (Longs en positif, Shorts en négatif)
        for col in df_weights.columns:
            if col != 'Cash':
                fig.add_trace(go.Bar(
                    x=df_weights.index,
                    y=df_weights[col],
                    name=col
                ))
        
        # 3. Affichage du Cash (Positif, au dessus de la ligne 0)
        if 'Cash' in df_weights.columns:
            fig.add_trace(go.Bar(
                x=df_weights.index,
                y=df_weights['Cash'],
                name='Cash (Non Investi)',
                marker_color='lightgrey'
            ))

        fig.update_layout(
            title="Allocation Dynamique (Exposition Nette)",
            xaxis_title="Date",
            yaxis_title="Exposition (1.0 = 100% Investi)",
            barmode='relative', # Les Shorts partent vers le bas, les Longs et le Cash vers le haut
            height=500,
            hovermode="x unified"
        )
        
        # On ajoute une ligne horizontale à 0 pour bien séparer Long/Short
        fig.add_hline(y=0, line_color="black", line_width=1)
        
        return fig
    
    def plot_macro_correlation(self, df_results, full_data, macro_ticker, macro_name, comparison_mode="Capital"):
        """
        Génère un graphique comparant la stratégie avec un indicateur Macro.
        Télécharge automatiquement les données si elles sont absentes.
        """
        import yfinance as yf
        from plotly.subplots import make_subplots
        import pandas as pd
        import plotly.graph_objects as go
        
        macro_series = None

        # 1. Tentative de récupération depuis full_data
        try:
            if macro_ticker in full_data.index.get_level_values('Ticker'):
                macro_series = full_data.xs(macro_ticker, level='Ticker')['Close']
        except:
            pass
            
        # 2. Si absent, on télécharge à la volée (Fallback)
        if macro_series is None or macro_series.empty:
            try:
                # On prend une marge de sécurité sur les dates
                start_d = df_results.index[0] - pd.Timedelta(days=10)
                end_d = df_results.index[-1] + pd.Timedelta(days=5)
                # Téléchargement silencieux
                data_dl = yf.download(macro_ticker, start=start_d, end=end_d, progress=False)
                if not data_dl.empty:
                    # Gestion du format multi-index ou simple de yfinance
                    if isinstance(data_dl.columns, pd.MultiIndex):
                        macro_series = data_dl['Close'].iloc[:, 0] # On prend la première colonne
                    else:
                        macro_series = data_dl['Close']
            except Exception as e:
                print(f"Erreur téléchargement {macro_ticker}: {e}")
                return None, 0.0

        if macro_series is None or macro_series.empty:
            return None, 0.0

        # 3. Alignement des données
        # Reindex permet de matcher exactement les dates du backtest
        macro_aligned = macro_series.reindex(df_results.index, method='ffill')

        # 4. Préparation Données Stratégie
        if comparison_mode == "Capital":
            if 'Strategy_Equity' not in df_results.columns:
                df_results['Strategy_Equity'] = self.capital * (1 + df_results['Net_Return']).cumprod()
            y_strat = df_results['Strategy_Equity']
            name_strat = "Ma Stratégie ($)"
            color_strat = '#2962FF'
        else:
            if 'Strategy_Equity' not in df_results.columns:
                df_results['Strategy_Equity'] = self.capital * (1 + df_results['Net_Return']).cumprod()
            cummax = df_results['Strategy_Equity'].cummax()
            dd = (df_results['Strategy_Equity'] - cummax) / cummax
            y_strat = dd
            name_strat = "Drawdown Stratégie"
            color_strat = '#D32F2F'

        # 5. Graphique Double Axe
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Trace Stratégie
        fig.add_trace(go.Scatter(x=df_results.index, y=y_strat, name=name_strat, line=dict(color=color_strat, width=2)), secondary_y=False)
        
        # Trace Macro
        fig.add_trace(go.Scatter(x=df_results.index, y=macro_aligned, name=f"{macro_name} (Axe Droit)", line=dict(color='orange', dash='dot', width=1.5), opacity=0.8), secondary_y=True)

        # 6. Corrélation
        strat_rets = df_results['Net_Return']
        macro_rets = macro_aligned.pct_change().fillna(0)
        correlation = strat_rets.corr(macro_rets)

        # Mise en forme
        fig.update_layout(
            title=" ", 
            height=450, 
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=40),
            legend=dict(orientation="h", y=1.1)
        )
        fig.update_yaxes(title_text=name_strat, secondary_y=False)
        if comparison_mode == "Drawdown": fig.update_yaxes(tickformat=".0%", secondary_y=False)
        
        return fig, correlation
    

    def plot_tranching_results(self, tranches_results):
        """
        Affiche les 4 courbes de Capital + Les Benchmarks standards.
        """
        import yfinance as yf
        
        colors = ['#2962FF', '#FF6D00', '#00C853', '#D500F9']
        
        # 1. Plage de dates globale
        all_dfs = [data['df'] for data in tranches_results.values()]
        start_d = min([df.index[0] for df in all_dfs])
        end_d = max([df.index[-1] for df in all_dfs])
        
        # 2. Téléchargement Benchmarks (S&P, Taux, Bonds)
        try:
            tickers_bench = ['BIL', 'TLT', '^GSPC']
            # Marge de 7 jours pour être sûr d'avoir les données
            bench_data = yf.download(tickers_bench, start=start_d - pd.Timedelta(days=7), end=end_d + pd.Timedelta(days=7), progress=False)['Close']
            
            # Alignement sur l'index de la première tranche (pour simplifier l'axe X)
            # On utilise ffill pour remplir les jours fériés éventuels
            ref_index = all_dfs[0].index
            bench_aligned = bench_data.reindex(ref_index, method='ffill').fillna(method='bfill')

            # Normalisation (Base 100 sur Capital Initial)
            risk_free_curve = (bench_aligned['BIL'] / bench_aligned['BIL'].iloc[0]) * self.capital
            sp500_curve = (bench_aligned['^GSPC'] / bench_aligned['^GSPC'].iloc[0]) * self.capital
            
            # 60/40
            bond_part = (bench_aligned['TLT'] / bench_aligned['TLT'].iloc[0]) * self.capital * 0.40
            equity_part = sp500_curve * 0.60
            balanced_60_40 = equity_part + bond_part
            
        except Exception as e:
            print(f"Benchmark Warning: {e}")
            ref_index = all_dfs[0].index
            risk_free_curve = pd.Series(self.capital, index=ref_index)
            sp500_curve = pd.Series(self.capital, index=ref_index)
            balanced_60_40 = pd.Series(self.capital, index=ref_index)

        # 3. Construction Graphique
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3],
                            subplot_titles=("Comparaison Stratégies vs Marché", "Drawdowns"))

        # --- A. BENCHMARKS (Fond) ---
        fig.add_trace(go.Scatter(x=ref_index, y=sp500_curve, name='S&P 500', line=dict(color='grey', dash='dot', width=1), opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=ref_index, y=balanced_60_40, name='60/40', line=dict(color='#3366CC', dash='dashdot', width=1), opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=ref_index, y=risk_free_curve, name='Cash', line=dict(color='orange', dash='dash', width=1), opacity=0.6), row=1, col=1)

        # --- B. LES 4 TRANCHES ---
        for i, (tranche_name, data) in enumerate(tranches_results.items()):
            df_res = data['df']
            c = colors[i % 4]
            
            # Capital
            fig.add_trace(go.Scatter(
                x=df_res.index, y=df_res['Capital'], 
                name=f"{tranche_name}",
                line=dict(color=c, width=2)
            ), row=1, col=1)
            
            # Drawdown
            equity = df_res['Capital']
            dd = (equity - equity.cummax()) / equity.cummax()
            fig.add_trace(go.Scatter(
                x=df_res.index, y=dd, 
                name=f"DD {tranche_name}",
                line=dict(color=c, width=1),
                showlegend=False
            ), row=2, col=1)

        fig.update_layout(height=650, title="Robustesse du Timing", hovermode="x unified")
        fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", tickformat=".0%", row=2, col=1)
        
        return fig

    def plot_tranching_costs(self, tranches_results):
        """ Compare les coûts cumulés des 4 stratégies """
        colors = ['#2962FF', '#FF6D00', '#00C853', '#D500F9']
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Frais Cumulés ($)", "% Frais Périodique (Lissé)"))

        for i, (tranche_name, data) in enumerate(tranches_results.items()):
            df_res = data['df']
            color = colors[i % len(colors)]
            
            # 1. Cumul
            cum_costs = df_res['Transaction_Costs'].cumsum()
            fig.add_trace(go.Scatter(
                x=df_res.index, y=cum_costs, 
                name=f"Frais {tranche_name}",
                line=dict(color=color, width=2)
            ), row=1, col=1)
            
            # 2. Pourcentage (Lissé par moyenne mobile pour lisibilité car 4 courbes)
            if 'Cost_Percent' in df_res.columns:
                pct = df_res['Cost_Percent'] * 100
            else:
                pct = (df_res['Transaction_Costs'] / df_res['Capital']) * 100
            
            # On affiche une ligne fine pour le %
            fig.add_trace(go.Scatter(
                x=df_res.index, y=pct,
                name=f"% {tranche_name}",
                line=dict(color=color, width=1, dash='dot'),
                showlegend=False
            ), row=2, col=1)

        fig.update_layout(height=500, title="Analyse des Coûts (Multi-Scénarios)", hovermode="x unified")
        fig.update_yaxes(title_text="Cumul ($)", row=1, col=1)
        fig.update_yaxes(title_text="% Capital", row=2, col=1)
        
        return fig

    def plot_tranching_allocations(self, tranches_results):
        """ 
        Affiche les allocations des 4 tranches avec :
        - Couleurs unifiées (Même actif = Même couleur partout)
        - Légende unique (Partagée pour tous les sous-graphiques)
        """
        import plotly.colors as pc
        
        num_tranches = len(tranches_results)
        row_titles = [f"Alloc. {name}" for name in tranches_results.keys()]
        
        # 1. Identifier tous les actifs présents (union de toutes les colonnes)
        all_assets = set()
        for data in tranches_results.values():
            if data['weights']:
                df_temp = pd.DataFrame(data['weights'])
                all_assets.update(df_temp.columns)
        
        # On retire 'Cash' s'il est calculé après, ou on le garde s'il est déjà là. 
        # Dans votre logique, 'Cash' est calculé dans la fonction, donc on liste les tickers.
        sorted_assets = sorted(list(all_assets))
        
        # 2. Création de la Palette de Couleurs Fixe
        # On utilise une palette qualitative large (ex: Dark24 ou Plotly) pour éviter les répétitions
        palette = pc.qualitative.Dark24 + pc.qualitative.Light24
        color_map = {asset: palette[i % len(palette)] for i, asset in enumerate(sorted_assets)}
        color_map['Cash'] = 'lightgrey' # Couleur imposée pour le Cash
        
        # 3. Création des Subplots
        fig = make_subplots(
            rows=num_tranches, cols=1, 
            shared_xaxes=True, 
            subplot_titles=row_titles,
            vertical_spacing=0.05
        )
        
        for i, (tranche_name, data) in enumerate(tranches_results.items()):
            weights_list = data['weights']
            if not weights_list: continue
            
            # Reconstruction du DF Weights pour cette tranche
            df_w = pd.DataFrame(weights_list)
            df_w.index = pd.to_datetime(df_w.index)
            
            # Calcul Cash (Exposition Nette)
            current_exposure = df_w.abs().sum(axis=1)
            df_w['Cash'] = (1.0 - current_exposure).clip(lower=0)
            
            # Tri des colonnes pour que l'ordre d'empilement soit toujours le même
            # On met le Cash à la fin ou au début selon préférence
            cols_to_plot = sorted([c for c in df_w.columns if c != 'Cash'])
            if 'Cash' in df_w.columns:
                cols_to_plot.append('Cash')
            
            # Ajout des traces
            for col in cols_to_plot:
                # On affiche la légende seulement sur le premier graphique (Tranche 0)
                # Mais grâce au legendgroup, cliquer dessus agira sur tous les graphiques
                show_legend = True if (i == 0) else False
                
                fig.add_trace(go.Bar(
                    x=df_w.index, 
                    y=df_w[col], 
                    name=col, 
                    marker_color=color_map.get(col, 'grey'), # Couleur forcée
                    legendgroup=col,      # Groupe par actif
                    showlegend=show_legend
                ), row=i+1, col=1)
                
        fig.update_layout(
            title="Comparaison Temporelle des Allocations (Légende Unifiée)", 
            height=300 * num_tranches, # Hauteur adaptative
            barmode='relative',
            hovermode="x unified"
        )
        
        return fig
    
    def plot_annual_stats_comparison(self, tranches_results):
        """
        Génère les graphiques annuels avec un calcul RIGOUREUX :
        1. Rendements : Performance de l'année N (vs Fin année N-1).
        2. Drawdowns : "Intra-Year Drawdown" -> On remet le compteur à zéro chaque 1er Janvier.
        """
        import yfinance as yf
        import plotly.graph_objects as go
        
        colors = ['#2962FF', '#FF6D00', '#00C853', '#D500F9'] 
        bench_color = '#37474F' 

        # 1. Nettoyage des index
        all_dfs = []
        for data in tranches_results.values():
            df = data['df'].copy()
            df.index = pd.to_datetime(df.index)
            all_dfs.append(df)
            
        if not all_dfs: return go.Figure(), go.Figure()

        start_d = min([df.index[0] for df in all_dfs])
        end_d = max([df.index[-1] for df in all_dfs])
        
        # --- 2. TÉLÉCHARGEMENT BENCHMARK ---
        bench_yearly_ret = pd.Series(dtype=float)
        bench_yearly_dd = pd.Series(dtype=float) # On va le remplir manuellement
        
        try:
            bench_data = yf.download(self.benchmark_ticker, start=start_d, end=end_d, progress=False)
            
            if isinstance(bench_data.columns, pd.MultiIndex):
                try: price_series = bench_data.xs('Close', level=0, axis=1)[self.benchmark_ticker]
                except: price_series = bench_data.iloc[:, 0]
            else:
                price_series = bench_data['Close']
            
            price_series = price_series.dropna()
            
            if not price_series.empty:
                # A. Rendement Annuel
                bench_resampled = price_series.resample('YE').last()
                bench_yearly_ret = bench_resampled.pct_change()
                
                # Correction 1ère année Benchmark
                if len(bench_yearly_ret) > 0 and pd.isna(bench_yearly_ret.iloc[0]):
                    first_val = price_series.iloc[0]
                    first_year_end = bench_resampled.iloc[0]
                    if first_val > 0:
                        bench_yearly_ret.iloc[0] = (first_year_end / first_val) - 1
                
                # B. Drawdown "Intra-Year" Benchmark
                # On groupe par année et on applique le calcul de DD sur chaque groupe indépendamment
                def calc_max_dd(series):
                    if series.empty: return 0.0
                    # On normalise la série pour qu'elle commence "neutre" ou on prend juste le max local
                    peak = series.cummax()
                    dd = (series - peak) / peak
                    return dd.min()

                # On applique cette fonction à chaque année
                bench_yearly_dd = price_series.groupby(price_series.index.year).apply(calc_max_dd)
                # On remet l'index en format Date (Fin d'année) pour que le graphique s'aligne
                # bench_yearly_dd.index est actuellement [2016, 2017...], on veut [2016-12-31, ...]
                bench_yearly_dd.index = pd.to_datetime([f"{y}-12-31" for y in bench_yearly_dd.index])
            
        except Exception as e:
            print(f"⚠️ Erreur Benchmark : {e}")

        # ==============================================================================
        # FIGURE 1 : RENDEMENTS (Inchangé car c'était déjà juste)
        # ==============================================================================
        fig_ret = go.Figure()

        if not bench_yearly_ret.empty:
            bench_yearly_ret = bench_yearly_ret.dropna()
            years_str = bench_yearly_ret.index.strftime('%Y')
            fig_ret.add_trace(go.Bar(x=years_str, y=bench_yearly_ret.values, name="Benchmark", marker_color=bench_color, hovertemplate="Benchmark: %{y:.2%}<extra></extra>"))

        for i, (tranche_name, df) in enumerate(zip(tranches_results.keys(), all_dfs)):
            yearly_ret = df['Capital'].resample('YE').last().pct_change()
            
            if len(yearly_ret) > 0 and pd.isna(yearly_ret.iloc[0]):
                first_year_end = df['Capital'].resample('YE').last().iloc[0]
                first_year_ret = (first_year_end / self.capital) - 1
                yearly_ret.iloc[0] = first_year_ret
            
            yearly_ret = yearly_ret.dropna()
            years_str = yearly_ret.index.strftime('%Y')
            
            fig_ret.add_trace(go.Bar(x=years_str, y=yearly_ret.values, name=tranche_name, marker_color=colors[i % 4], opacity=0.85, hovertemplate=f"{tranche_name}: %{{y:.2%}}<extra></extra>"))

        fig_ret.update_layout(title="<b>📊 Rendements par Année Civile</b>", yaxis_title="Rendement (%)", template="plotly_white", barmode='group', bargap=0.15, height=500, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
        fig_ret.update_yaxes(tickformat=".1%")

        # ==============================================================================
        # FIGURE 2 : DRAWDOWN INTRA-YEAR (CORRIGÉ)
        # ==============================================================================
        fig_dd = go.Figure()

        # A. Benchmark
        if not bench_yearly_dd.empty:
            years_str = bench_yearly_dd.index.strftime('%Y')
            fig_dd.add_trace(go.Bar(
                x=years_str, 
                y=bench_yearly_dd.values,
                name="Benchmark", 
                marker_color=bench_color,
                hovertemplate="Bench Max DD: %{y:.2%}<extra></extra>"
            ))

        # B. Stratégies
        for i, (tranche_name, df) in enumerate(zip(tranches_results.keys(), all_dfs)):
            equity = df['Capital']
            
            # Fonction locale pour calculer le DD sur un sous-ensemble (une année)
            def calc_max_dd_local(series):
                if series.empty: return 0.0
                peak = series.cummax()
                dd = (series - peak) / peak
                return dd.min()

            # GroupBy Year -> Apply Drawdown Calc
            yearly_dd = equity.groupby(equity.index.year).apply(calc_max_dd_local)
            
            # Reconstruction des dates pour l'axe X (Années)
            # On s'assure de l'alignement avec les rendements
            years_str = [str(y) for y in yearly_dd.index]
            
            fig_dd.add_trace(go.Bar(
                x=years_str, 
                y=yearly_dd.values,
                name=tranche_name, 
                marker_color=colors[i % 4],
                opacity=0.85,
                hovertemplate=f"{tranche_name} Max DD: %{{y:.2%}}<extra></extra>"
            ))

        fig_dd.update_layout(
            title="<b>📉 Pire Baisse Intra-Annuelle (Remise à zéro au 1er Janv)</b>",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            barmode='group',
            bargap=0.15,
            height=500,
            hovermode="x unified"
        )
        fig_dd.update_yaxes(tickformat=".1%")

        return fig_ret, fig_dd