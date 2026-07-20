import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Importation de vos modules locaux
from src.data_loader import DataLoader
from src.features import FeatureEngineer
# On importe XGBoostModeler au lieu de RegressionModels
from src.models import ModelPreparator, XGBoostModeler, StatisticalModeler, ASSETS_CONFIG
from src.strategy import Strategy
from sklearn.metrics import mean_squared_error, accuracy_score

# Configuration de la page
st.set_page_config(
    page_title="Stratégie Directionnelle & ML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration pour le téléchargement Haute Définition
config_hd = {
    'displayModeBar': True,
    'scrollZoom': True,
    'toImageButtonOptions': {
        'format': 'svg',        # FORMAT VECTORIEL (Le meilleur pour Canva/PPT)
        'filename': 'analyse_finance_hd',
        'height': 1080,         # Hauteur en pixels (ex: 1080p)
        'width': 1920,          # Largeur en pixels
        'scale': 3              # Multiplicateur de qualité (3x = Ultra Net)
    }
}

# Portefeuille Utilisateur
USER_PORTFOLIO_2 = [
    # '^IXIC',   # Nasdaq
     '^GSPC',   # S&P 500 
     'TLT',     # Bonds 20y
    # 'VNQ',     # Real Estate
     'GLD',     # Or
    # 'EURUSD=X', # Euro/Dollar
    # '^FCHI',  # CAC40
    # 'USO'  # Pétrole
]
# '^IXIC',   # Nasdaq

USER_PORTFOLIO = [
    'XLK',  # Technologie
    'XLF',  # Finance
    'XLV',  # Santé
    'XLY',  # Conso Discrétionnaire
    'XLP',  # Conso Base
    'XLE',  # Énergie
    # 'XLC',  # Communication too young
    'XLI',  # Industrie
    'XLB',  # Matériaux
    'XLU',  # Utilities
    # 'XLRE' # Immobilier too young
]

# Portefeuille Utilisateur (Indices Actions + ETFs pour le reste)
USER_PORTFOLIO_1 = [
    '^GSPC',      # S&P 500
    '^FCHI',      # CAC 40
    '^GDAXI',     # DAX
    '^FTSE',      # FTSE 100
    '^BSESN',     # SENSEX
    'ASHR',       # Chine (ETF CSI 300 - Plus fiable que 000010.SS sur Yahoo)
    'GLD',        # Or (ETF)
    'USO',        # Pétrole (ETF)
    'TLT',        # Obligations US 20+ ans (ETF)
    'VNQ',        # Immobilier US (ETF)
    'BTC-USD'     # Bitcoin
]

# --- FONCTIONS UTILITAIRES & CACHE ---

@st.cache_data
def get_full_data():
    """
    Charge les données, applique le Feature Engineering et prépare les rendements.
    """
    tickers_2 = ['^GSPC', '^IXIC', '^FCHI', 'TLT', 'GLD', 'EURUSD=X', 'USO', 'VNQ']
    tickers = [
    '^GSPC',  # S&P 500
    'XLK',  # Technologie
    'XLF',  # Finance
    'XLV',  # Santé
    'XLY',  # Conso Discrétionnaire
    'XLP',  # Conso Base
    'XLE',  # Énergie
    # 'XLC',  # Communication too young
    'XLI',  # Industrie
    'XLB',  # Matériaux
    'XLU',  # Utilities
    # 'XLRE' # Immobilier too young
    ]
    tickers1 = ['^GSPC', '^FCHI', '^GDAXI', '^FTSE', '^BSESN', 'ASHR', 'GLD', 'USO', 'TLT', 'VNQ', 'BTC-USD']
    
    loader = DataLoader(tickers=tickers, period="20y")
    raw_data = loader.load_data()
    
    fe = FeatureEngineer(raw_data)
    df_features = fe.add_moving_average(window=5, title='SMA_5')
    df_features = fe.add_moving_average(window=13, title='SMA_13')
    df_features = fe.add_moving_average(window=20, title='SMA_20')
    df_features = fe.add_moving_average(window=26, title='SMA_26')
    df_features = fe.add_moving_average(window=50, title='SMA_50')
    df_features = fe.add_moving_average(window=200, title='SMA_200')
    df_features = fe.add_rsi(window=14)
    df_features = fe.add_lags(lags=[1, 2, 3, 5])
    # 2. Bollinger (Volatilité - Crucial pour Pétrole/Commodities)
    df_features = fe.add_bollinger_bands(window=20, num_std=2)   
    # 3. MACD (Momentum - Crucial pour CAC40/Indices)
    df_features = fe.add_macd()
    # 4. Contexte Macroéconomique (VIX, Taux, Dollar Index)
    macro_tickers = {
                    'VIX': '^VIX',       # Peur / Confiance
                    'Rates': '^TNX',     # Taux 10 ans
                    'DXY': 'DX-Y.NYB',   # Dollar Index (Fort impact Pétrole)
                    'OilVol': '^OVX'     # Volatilité Pétrole (Spécifique)
                }
                
    # On télécharge les données macro sur la même période (max)
    with st.spinner("Téléchargement des indicateurs macro..."):
        macro_loader = DataLoader(tickers=list(macro_tickers.values()), period="20y")
        macro_data = macro_loader.load_data()
                
    # --- INJECTION MACRO ---
    # 1. VIX (Confiance Globale)
    try:
        vix_data = macro_data.xs(macro_tickers['VIX'], level='Ticker')
        df_features = fe.add_context(vix_data, prefix='VIX')
    except: pass # Si échec téléchargement
                
    # 2. Taux (Macro Globale)
    try:
        tnx_data = macro_data.xs(macro_tickers['Rates'], level='Ticker')
        df_features = fe.add_context(tnx_data, prefix='Rates')
    except: pass

    try:
        dxy_data = macro_data.xs(macro_tickers['DXY'], level='Ticker')
        df_features = fe.add_context(dxy_data, prefix='DXY')
    except: pass
    # 4. Volatilité Pétrole (Pour Pétrole uniquement)
    #try:
    #    ovx_data = macro_data.xs(macro_tickers['OilVol'], level='Ticker')
    #    df_features = fe.add_context(ovx_data, prefix='OilVol')
    #except: pass
                
    
    strat = Strategy(df_features)
    df_regime = strat.define_market_regime(col_ma_short='SMA_50', col_ma_long='SMA_200')
    
    prep = ModelPreparator(df_regime)
    df_final = prep.prepare_data()
    
    return df_final

# --- Fonction pour mettre arrière-plan image transparente ---

def make_transparent(fig):
    """ 
    Prépare le graphique pour l'export :
    1. Fond transparent
    2. Force TOUT en NOIR (Titres, Axes, Légendes, Grilles)
    """
    # 1. Mise à jour globale (Fond + Texte général + Légende)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {'color': 'white'},  # Couleur par défaut
        # 'title': {'font': {'color': 'white'}},
        'legend': {
            'font': {'color': 'white'},
            'bgcolor': 'rgba(0,0,0,0)' # Légende transparente aussi
        }
    })

    # 2. Forcer spécifiquement les AXES X (Absisses)
    # On utilise update_xaxes qui s'applique à tous les sous-graphiques éventuels
    fig.update_xaxes(
        title_font=dict(color='white'),  # Couleur du Titre de l'axe
        tickfont=dict(color='white'),    # Couleur des Chiffres/Dates
        showline=True, linecolor='white', # Couleur de la ligne de l'axe (le trait)
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)', # Grille légère
        zeroline=True, zerolinecolor='rgba(0, 0, 0, 0.2)' # Ligne du zéro un peu plus visible
    )

    # 3. Forcer spécifiquement les AXES Y (Ordonnées)
    fig.update_yaxes(
        title_font=dict(color='white'),
        tickfont=dict(color='white'),
        showline=True, linecolor='white',
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
        zeroline=True, zerolinecolor='rgba(0, 0, 0, 0.2)'
    )

    return fig

# --- NAVIGATION ---

st.sidebar.title("Navigation")
pages = [
    "1. Analyse de Données & Indicateurs", 
    "2. Modélisation (GARCH & ARIMA-XGBoost)", 
    "3. Backtesting & Performance",
    "4. Live Trading Dashboard"
]
selection = st.sidebar.radio("Aller vers :", pages)

# Chargement des données global
try:
    with st.spinner('Chargement des données et calcul des indicateurs...'):
        df = get_full_data()
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()

# --- PAGE 1 : ANALYSE DE DONNEES ---

if selection == "1. Analyse de Données & Indicateurs":
    st.title("📊 Terminal d'Analyse Technique")
    st.markdown("Visualisation professionnelle multi-indicateurs (Style ProRealTime).")

    # --- BARRE LATÉRALE : SÉLECTION DES INDICATEURS (Commun à la page) ---
    st.sidebar.header("🔧 Configuration Graphique")
    
    # 1. Choix de l'actif
    available_tickers = df.index.get_level_values('Ticker').unique()
    display_list = [t for t in USER_PORTFOLIO if t in available_tickers]
    selected_ticker = st.sidebar.selectbox("Actif", display_list)
    
    # 2. Indicateurs de Prix (Overlay)
    st.sidebar.subheader("Sur le Prix")
    show_sma = st.sidebar.multiselect(
        "Moyennes Mobiles", 
        ['SMA_13', 'SMA_26', 'SMA_50', 'SMA_200'], 
        default=['SMA_50', 'SMA_200']
    )
    show_bollinger = st.sidebar.checkbox("Bandes de Bollinger", value=False)
    regime_method = st.sidebar.radio(
        "Détection de Régime (Fond Coloré)", 
        ["Aucun", "SMA 50/200 (Classique)", "HMM (Statistique/IA)"],
        index=1
    )

    # 3. Indicateurs Secondaires (Sous-Graphiques)
    st.sidebar.subheader("Oscillateurs & Volume")
    show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=False)
    show_returns = st.sidebar.checkbox("Log Returns (Barres)", value=True)

    # --- CRÉATION DES ONGLETS ---
    tab_tech, tab_corr = st.tabs(["📈 Analyse Technique", "🔗 Matrice de Corrélation"])

    # ==============================================================================
    # ONGLET 1 : ANALYSE TECHNIQUE (MODIFIÉ POUR HMM PROBABILITY)
    # ==============================================================================
    with tab_tech:
        # --- PRÉPARATION DES DONNÉES ---
        df_asset = df.xs(selected_ticker, level='Ticker').copy()
        
        # --- LOGIQUE DE RÉGIME DYNAMIQUE (SMA vs HMM) ---
        df_asset['Regime_Visu'] = 0 # Par défaut neutre
        
        # Flag pour savoir si on doit afficher le graph de probabilité
        show_hmm_prob = False 
        
        if regime_method == "SMA 50/200 (Classique)":
            # Ancienne méthode
            cond_bull = (df_asset['Close'] > df_asset['SMA_50']) & (df_asset['SMA_50'] > df_asset['SMA_200'])
            cond_bear = (df_asset['Close'] < df_asset['SMA_50']) & (df_asset['SMA_50'] < df_asset['SMA_200'])
            df_asset.loc[cond_bull, 'Regime_Visu'] = 1
            df_asset.loc[cond_bear, 'Regime_Visu'] = -1
            
        elif regime_method == "HMM (Statistique/IA)":
            # Nouvelle méthode HMM
            from src.strategy import HMMStrategy 
            
            # Instanciation et calcul
            hmm_strat = HMMStrategy(df_asset, n_states=2)
            df_hmm = hmm_strat.fit_predict(vol_window=12)
            
            # Jointure via l'index Date
            df_asset['Regime_Visu'] = df_hmm['HMM_Regime']
            
            # --- AJOUT CRUCIAL : Récupération de la probabilité ---
            if 'Bear_Prob' in df_hmm.columns:
                df_asset['Bear_Prob'] = df_hmm['Bear_Prob']
                show_hmm_prob = True # On active l'affichage du sous-graphique
            
            # On remplit les trous
            df_asset['Regime_Visu'] = df_asset['Regime_Visu'].fillna(0)
            if show_hmm_prob:
                df_asset['Bear_Prob'] = df_asset['Bear_Prob'].fillna(0)
            
            st.caption("ℹ️ **Info HMM** : Le modèle détecte les régimes basés sur la Volatilité et les Rendements.")

        # --- CONSTRUCTION DU GRAPHIQUE (SUBPLOTS) ---
        rows_specs = [{'type': 'xy'}] 
        row_titles = [f"Cours : {selected_ticker}"]
        
        # Gestion dynamique des hauteurs
        added_rows = 0
        
        # 1. AJOUT DU SLOT POUR LA PROBA HMM (Prioritaire, juste sous le prix)
        if show_hmm_prob:
            rows_specs.append({'type': 'xy'})
            row_titles.append("Probabilité de Krach P(Bear)")
            added_rows += 1
            
        # 2. Autres indicateurs
        if show_rsi: 
            rows_specs.append({'type': 'xy'})
            row_titles.append("RSI (14)")
            added_rows += 1
        if show_macd:
            rows_specs.append({'type': 'xy'})
            row_titles.append("MACD")
            added_rows += 1
        if show_returns:
            rows_specs.append({'type': 'xy'})
            row_titles.append("Rendements Hebdo (%)")
            added_rows += 1
            
        # Recalcul des hauteurs relatives
        if added_rows > 0:
            main_h = 0.5 # Le prix prend 50% de l'espace
            sub_h = 0.5 / added_rows # Le reste est partagé
            row_heights = [main_h] + [sub_h] * added_rows
        else:
            row_heights = [1.0]

        fig = make_subplots(
            rows=1 + added_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=row_titles
        )

        # === ROW 1 : PRIX & OVERLAYS ===
        current_row = 1
        
        # 1. Chandelier
        fig.add_trace(go.Candlestick(
            x=df_asset.index,
            open=df_asset['Open'], high=df_asset['High'],
            low=df_asset['Low'], close=df_asset['Close'],
            name='Prix',
            increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
        ), row=current_row, col=1)

        # 2. Moyennes Mobiles
        colors_sma = {'SMA_13': 'yellow', 'SMA_26': 'orange', 'SMA_50': 'blue', 'SMA_200': 'purple'}
        for sma in show_sma:
            if sma in df_asset.columns:
                fig.add_trace(go.Scatter(
                    x=df_asset.index, y=df_asset[sma],
                    name=sma,
                    line=dict(color=colors_sma.get(sma, 'grey'), width=1)
                ), row=current_row, col=1)

        # 3. Bandes de Bollinger
        if show_bollinger and 'BB_High' in df_asset.columns:
            fig.add_trace(go.Scatter(
                x=df_asset.index, y=df_asset['BB_High'],
                line=dict(width=0), showlegend=False, name='BB High'
            ), row=current_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_asset.index, y=df_asset['BB_Low'],
                line=dict(width=0), fill='tonexty', 
                fillcolor='rgba(0, 100, 250, 0.1)', 
                showlegend=True, name='Bollinger'
            ), row=current_row, col=1)

        # 4. Régime de Marché (Fond Coloré sur le Graphique PRIX)
        if regime_method != "Aucun":
            regime_changes = df_asset['Regime_Visu'].diff().fillna(0)
            change_indices = regime_changes[regime_changes != 0].index
            
            if len(df_asset) > 0:
                full_indices = sorted(list(set([df_asset.index[0]] + list(change_indices) + [df_asset.index[-1]])))
                
                shapes = []
                for i in range(len(full_indices) - 1):
                    start_date = full_indices[i]
                    end_date = full_indices[i+1]
                    try: 
                        val = df_asset.loc[start_date, 'Regime_Visu']
                    except: continue
                        
                    if val == 1: color = "rgba(0, 255, 0, 0.08)"   # Vert
                    elif val == -1: color = "rgba(255, 0, 0, 0.08)" # Rouge
                    else: continue
                    
                    shapes.append(dict(
                        type="rect", xref="x", yref="paper", # paper ref pour couvrir le subplot
                        x0=start_date, x1=end_date, y0=0, y1=1,
                        fillcolor=color, layer="below", line_width=0,
                    ))
                # On applique les formes uniquement au layout global (elles se placent selon x/y ref)
                # Note: yref='paper' couvre tout en hauteur, ce qui est ok pour le fond
                fig.update_layout(shapes=shapes)

        # === ROW : PROBABILITÉ HMM (SI ACTIVÉ) ===
        if show_hmm_prob:
            current_row += 1
            # Zone Rouge remplie
            fig.add_trace(go.Scatter(
                x=df_asset.index, y=df_asset['Bear_Prob'],
                name='Proba Krach',
                mode='lines',
                line=dict(color='red', width=1),
                fill='tozeroy', # Remplit jusqu'à 0
                fillcolor='rgba(255, 0, 0, 0.2)'
            ), row=current_row, col=1)
            
            # Ligne de seuil 50%
            fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.update_yaxes(range=[0, 1], title_text="Prob.", row=current_row, col=1)

        # === ROW SUIVANTS : INDICATEURS CLASSIQUES ===
        if show_rsi and 'RSI' in df_asset.columns:
            current_row += 1
            fig.add_trace(go.Scatter(
                x=df_asset.index, y=df_asset['RSI'],
                name='RSI', line=dict(color='purple', width=1.5)
            ), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)

        if show_macd and 'MACD' in df_asset.columns:
            current_row += 1
            fig.add_trace(go.Scatter(
                x=df_asset.index, y=df_asset['MACD'],
                name='MACD', line=dict(color='blue', width=1)
            ), row=current_row, col=1)
            fig.add_trace(go.Scatter(
                x=df_asset.index, y=df_asset['MACD_Signal'],
                name='Signal', line=dict(color='orange', width=1)
            ), row=current_row, col=1)
            colors_macd = np.where(df_asset['MACD_Hist'] >= 0, 'green', 'red')
            fig.add_trace(go.Bar(
                x=df_asset.index, y=df_asset['MACD_Hist'],
                name='Hist', marker_color=colors_macd
            ), row=current_row, col=1)

        if show_returns:
            current_row += 1
            rets = df_asset['Log Returns'] * 100 
            colors_ret = np.where(rets >= 0, '#26A69A', '#EF5350')
            fig.add_trace(go.Bar(
                x=df_asset.index, y=rets,
                name='Rendement %', marker_color=colors_ret
            ), row=current_row, col=1)

        # === LAYOUT FINAL ===
        fig.update_layout(
            height=300 + (added_rows * 200),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            template="plotly_white"
        )
        # Fonction helper supposée existante ou simple appel
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats descriptives
        st.markdown("### 📋 Statistiques de l'Actif")
        stats = df_asset['Log Returns'].describe()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volatilité Hebdo", f"{stats['std']*100:.2f}%")
        c2.metric("Rendement Max", f"{stats['max']*100:.2f}%")
        c3.metric("Rendement Min", f"{stats['min']*100:.2f}%")
        c4.metric("Dernier Close", f"{df_asset['Close'].iloc[-1]:.2f}")

    # ==============================================================================
    # ONGLET 2 : MATRICE DE CORRÉLATION (NOUVEAU)
    # ==============================================================================
    with tab_corr:
        st.subheader("Corrélation entre les actifs du portefeuille")
        st.markdown("Calculé sur la base des **Rendements Logarithmiques** historiques.")
        
        # 1. Pivot pour avoir Dates en Index et Tickers en Colonnes
        # On filtre d'abord pour ne garder que les actifs du portefeuille utilisateur
        df_portfolio = df[df.index.get_level_values('Ticker').isin(USER_PORTFOLIO)]
        
        # On pivote
        corr_data = df_portfolio.pivot_table(index='Date', columns='Ticker', values='Log Returns')
        
        # 2. Calcul de la matrice
        corr_matrix = corr_data.corr()
        
        # 3. Affichage Heatmap avec Plotly
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            zmin=-1, zmax=1,
            colorscale='RdBu', # Rouge = Négatif, Bleu = Positif (Standard Finance)
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            showscale=True
        ))
        
        fig_corr.update_layout(
            title="Matrice de Corrélation (Heatmap)",
            height=600,
            template="plotly_white"
        )
        
        st.plotly_chart(make_transparent(fig_corr), use_container_width=True)

# --- PAGE 2 : MODELISATION (GARCH & XGBoost) ---

elif selection == "2. Modélisation (GARCH & ARIMA-XGBoost)":
    st.title("🤖 Modélisation Prédictive")
    
    tab1, tab2 = st.tabs(["Volatilité (GARCH)", "Rendements (ARIMA-XGBoost)"])
    
    # --- TAB 1 : GARCH ---
    with tab1:
        st.header("Modélisation Statistique (Séries Temporelles)")
        st.markdown("""
        Cette section utilise une approche classique : **ARIMA** pour la tendance (rendement moyen) 
        et **GARCH** pour le risque (volatilité).
        """)

        # 1. SÉLECTEURS
        # On passe à 3 colonnes pour ajouter le Lookback
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        
        with col_sel1:
            tickers_list = df.index.get_level_values('Ticker').unique()
            display_list_1 = [t for t in USER_PORTFOLIO if t in tickers_list]
            selected_ticker_stat = st.selectbox("Choisir l'actif à analyser", display_list_1, key="stat_ticker")
        
        with col_sel2:
            train_size_pct = st.slider("Split Train/Test (%)", 50, 95, 80, key="stat_slider") / 100.0

        with col_sel3:
            lookback_garch = st.selectbox(
                "Profondeur d'entraînement (Walk-Forward)", 
                ["Tout l'historique", "1 An", "2 Ans", "3 Ans", "5 Ans"], 
                index=0, 
                key="garch_lookback",
                help="Fenêtre glissante utilisée pour le ré-entraînement du modèle à chaque pas de temps."
            )

        if st.button("Lancer l'analyse GARCH", type="primary"):
            
            # 2. PRÉPARATION DES DONNÉES
            try:
                # A. Extraction
                mask = df.index.get_level_values('Ticker') == selected_ticker_stat
                df_asset = df.loc[mask].copy()
                
                if df_asset.empty:
                    st.error(f"Aucune donnée trouvée pour {selected_ticker_stat}.")
                    st.stop()

                # B. Split (Static)
                prep = ModelPreparator(df_asset)
                train_df, test_df = prep.train_test_split(train_size=train_size_pct)
                
                train_returns = train_df['Log Returns'].astype(float).dropna()
                test_returns = test_df['Log Returns'].astype(float).dropna()
                
                if len(train_returns) < 30:
                    st.error("Pas assez de données d'entraînement (moins de 30 points).")
                    st.stop()

                # Dates pour les graphs
                train_dates = train_returns.index.get_level_values('Date')
                test_dates = test_returns.index.get_level_values('Date')

                # --- MODELING (Approche Statique) ---
                with st.spinner('Analyse Statique en cours...'):
                    modeler = StatisticalModeler(train_returns)
                    
                    # 1. Fit ARIMA
                    arima_model = modeler.fit_arima()
                    if arima_model is None:
                        st.error("Échec de l'entraînement ARIMA.")
                        st.stop()

                    # 2. Fit GARCH (Static)
                    garch_results = modeler.fit_garch()
                    
                    # 3. Prédictions Statiques (Benchmark)
                    n_forecast = len(test_returns)              
                    
                    # Volatilité
                    if garch_results is not None:
                        garch_forecast = garch_results.forecast(horizon=n_forecast)
                        pred_vol_variance = garch_forecast.variance.iloc[-1].values 
                        pred_vol_series = pd.Series(np.sqrt(pred_vol_variance), index=test_dates)
                    else:
                        pred_vol_series = pd.Series(0, index=test_dates)

                    # --- Construction de la "Vérité" (Volatilité Ex-Post sur tout l'échantillon) ---
                    # On utilise le modèle ajusté sur tout l'historique pour estimer quelle était la "vraie" volatilité
                    full_series = pd.concat([train_returns, test_returns])
                    modeler_test = StatisticalModeler(full_series)
                    modeler_test.fit_arima()
                    garch_results_test = modeler_test.fit_garch()
                    # On récupère la volatilité conditionnelle calculée par le modèle sur la partie Test
                    test_fitted_vol = garch_results_test.conditional_volatility[-n_forecast:]

                # --- AFFICHAGE STATIQUE ---
                # Calcul des Métriques (Statique)
                rmse_vol_static = np.sqrt(mean_squared_error(test_fitted_vol, pred_vol_series))
                mae_vol_static = mean_absolute_error(test_fitted_vol, pred_vol_series)

                st.markdown("### 📊 Résultats (Approche Statique)")
                c1, c2 = st.columns(2)
                c1.metric("RMSE (Statique)", f"{rmse_vol_static:.3f}%")
                c2.metric("MAE (Statique)", f"{mae_vol_static:.3f}%")

                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=train_dates, y=garch_results.conditional_volatility, name='Volatilité Historique (Train)', line=dict(color='orange', width=1)))
                fig_vol.add_trace(go.Scatter(x=test_dates, y=pred_vol_series, name='Volatilité Prévue (Fixe)', line=dict(color='red', width=2)))
                fig_vol.add_trace(go.Scatter(x=test_dates, y=test_fitted_vol, name='Volatilité Réelle (Ex-Post)', line=dict(color='grey', width=1, dash='dot')))
                fig_vol.update_layout(title="Volatilité : Prévision Statique vs Réalité", height=400, template="plotly_white")
                st.plotly_chart(fig_vol, use_container_width=True)

                # =========================================================
                # 4. ANALYSE WALK-FORWARD (NOUVEAU)
                # =========================================================
                st.markdown("---")
                st.subheader("🔁 Analyse Walk-Forward (Volatilité Dynamique)")
                st.caption(f"Le modèle GARCH est ré-entraîné à chaque pas de temps sur une fenêtre glissante ({lookback_garch}).")

                wf_progress = st.progress(0)
                status_wf = st.empty()
                
                wf_preds_vol = []
                full_data_sorted = full_series.sort_index() # On a besoin de Train + Test
                
                # Boucle sur le Test Set
                for i, date in enumerate(test_dates):
                    # Progression
                    prog = (i + 1) / len(test_dates)
                    wf_progress.progress(prog)
                    
                    # A. Définition de la fenêtre d'entraînement
                    current_date = pd.to_datetime(date)
                    
                    # Cela renvoie une liste de Timestamps comparables
                    idx_dates = full_data_sorted.index.get_level_values('Date')
                    
                    if lookback_garch == "Tout l'historique":
                        # Expanding Window : Du début jusqu'à hier
                        mask_train = idx_dates < current_date
                    else:
                        # Rolling Window : De (Auj - X ans) jusqu'à hier
                        years = int(lookback_garch.split()[0])
                        start_date_wf = current_date - pd.DateOffset(years=years)
                        
                        mask_train = (idx_dates < current_date) & (idx_dates >= start_date_wf)
                    
                    train_window = full_data_sorted.loc[mask_train]
                    
                    # Sécurité
                    if len(train_window) < 30:
                        wf_preds_vol.append(0.0) # Fallback
                        continue
                        
                    try:
                        # B. Entraînement GARCH sur la fenêtre locale
                        # Note: StatisticalModeler s'attend à des séries pandas
                        mwf = StatisticalModeler(train_window)
                        mwf.fit_arima() # Nécessaire pour initialiser les résidus
                        res_wf = mwf.fit_garch()
                        
                        if res_wf:
                            # Prévision T+1
                            fc = res_wf.forecast(horizon=1)
                            # On récupère la variance prédite pour le dernier point
                            next_var = fc.variance.iloc[-1].values[0]
                            wf_preds_vol.append(np.sqrt(next_var))
                        else:
                            wf_preds_vol.append(0.0)
                            
                    except Exception:
                        wf_preds_vol.append(0.0)
                
                status_wf.empty()
                
                # Création de la série Walk-Forward
                wf_vol_series = pd.Series(wf_preds_vol, index=test_dates)
                
                # Calcul des Métriques (Walk-Forward)
                # On compare toujours à la "Volatilité Réelle (Ex-Post)" calculée plus haut
                rmse_wf = np.sqrt(mean_squared_error(test_fitted_vol, wf_vol_series))
                mae_wf = mean_absolute_error(test_fitted_vol, wf_vol_series)
                
                # Affichage Métriques
                c_wf1, c_wf2 = st.columns(2)
                c_wf1.metric("RMSE (Walk-Forward)", f"{rmse_wf:.3f}%", delta=f"{(rmse_vol_static - rmse_wf):.3f} vs Statique")
                c_wf2.metric("MAE (Walk-Forward)", f"{mae_wf:.3f}%", delta=f"{(mae_vol_static - mae_wf):.3f} vs Statique")
                
                # Graphique Walk-Forward
                fig_wf = go.Figure()
                
                # 1. Historique
                fig_wf.add_trace(go.Scatter(
                    x=train_dates, y=garch_results.conditional_volatility, 
                    name='Historique (Train)', 
                    line=dict(color='orange', width=1)
                ))
                
                # 2. Réalité (Target)
                fig_wf.add_trace(go.Scatter(
                    x=test_dates, y=test_fitted_vol, 
                    name='Volatilité Réelle (Ex-Post)', 
                    line=dict(color='grey', width=1, dash='dot')
                ))
                
                # 3. Prédiction Walk-Forward
                fig_wf.add_trace(go.Scatter(
                    x=test_dates, y=wf_vol_series, 
                    name='Prédiction Walk-Forward', 
                    line=dict(color='#00C853', width=2)
                ))
                
                fig_wf.update_layout(
                    title=f"Walk-Forward GARCH: {selected_ticker_stat} (Window: {lookback_garch})",
                    yaxis_title="Volatilité (%)",
                    xaxis_title="Date",
                    height=450,
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                if 'make_transparent' in globals():
                    st.plotly_chart(make_transparent(fig_wf), use_container_width=True)
                else:
                    st.plotly_chart(fig_wf, use_container_width=True)

            except Exception as e:
                st.error(f"Une erreur inattendue est survenue : {e}")
                st.exception(e)

    # --- TAB 2 : MODELE HYBRIDE (ARIMA + XGBOOST) ---
    with tab2:
        st.header("Modèle Hybride : ARIMA + XGBoost")
        st.markdown("""
        Ce modèle combine deux approches pour maximiser la performance :
        1.  **ARIMA** capture la **tendance linéaire** et l'autocorrélation simple.
        2.  **XGBoost** intervient ensuite pour **corriger les erreurs (résidus)** d'ARIMA en utilisant des indicateurs techniques avancés (RSI, FracDiff, Lags).
        """)

        # 1. SÉLECTEURS
        col_hyb1, col_hyb2, col_hyb3, col_hyb4 = st.columns(4)
        with col_hyb1:
            # Filtre sur le portefeuille utilisateur
            tickers_list_hyb = df.index.get_level_values('Ticker').unique()
            display_list_hyb = [t for t in USER_PORTFOLIO if t in tickers_list_hyb]
            selected_ticker_hyb = st.selectbox("Actif Cible", display_list_hyb, key="hyb_ticker")
        
        #with col_hyb2:
            #train_size_hyb = st.slider("Split Train/Test (%)", 50, 95, 80, key="hyb_slider") / 100.0
        
        with col_hyb2:
                
                # Le Widget Calendrier
                split_date_input = st.date_input(
                    "📅 Date de début du Test Set",
                    value="2025-01-01",
                    min_value="2005-01-01",
                    max_value=pd.to_datetime("today").date(),
                    help="Toutes les données AVANT cette date serviront à entraîner le modèle. Les données À PARTIR de cette date serviront au test."
                )
            
        with col_hyb3:
            # Petit réglage optionnel pour XGBoost
            n_estim = st.selectbox("Arbres XGBoost", [20, 50, 100, 200, 250, 500], index=0, key="hyb_est")
        
        with col_hyb4:
            lookback_option = st.selectbox(
                "Profondeur d'entraînement", 
                ["Tout l'historique", "1 An", "2 Ans", "3 Ans", "5 Ans"], 
                index=0, 
                key="hyb_lookback",
                help="Limite l'entraînement aux X dernières années avant la date de test."
            )

        if st.button("Lancer la Prédiction Hybride", type="primary"):
            
            status_container = st.empty()
            try:
                # =========================================================
                # 1. PRÉPARATION IDENTIQUE AU NOTEBOOK (Single Asset Fix)
                # =========================================================
                status_container.info("Phase 1/4 : Feature Engineering & FracDiff...")
                
                # A. Isolation de l'actif (Copie propre pour ne pas toucher au cache global)
                # On utilise .xs pour récupérer uniquement les données de l'actif choisi
                df_asset = df.xs(selected_ticker_hyb, level='Ticker', drop_level=False).copy()
                
                macro_tickers = {
                    'VIX': '^VIX',       # Peur / Confiance
                    'Rates': '^TNX',     # Taux 10 ans
                    'DXY': 'DX-Y.NYB',   # Dollar Index (Fort impact Pétrole)
                    'OilVol': '^OVX',     # Volatilité Pétrole (Spécifique)
                    'Silver': 'SI=F'     # Argent (Corrélé Or)
                }
                
                # On télécharge les données macro sur la même période (max)
                with st.spinner("Téléchargement des indicateurs macro..."):
                    macro_loader = DataLoader(tickers=list(macro_tickers.values()), period="20y")
                    macro_data = macro_loader.load_data()

                # B. Feature Engineering Spécifique
                fe_local = FeatureEngineer(df_asset)
                # 3. Dollar (Pour Pétrole/Or)
                if selected_ticker_hyb in ['USO', 'GLD']:
                    try:
                        dxy_data = macro_data.xs(macro_tickers['DXY'], level='Ticker')
                        df_asset = fe_local.add_context(dxy_data, prefix='DXY')
                        #sxy_data = macro_data.xs(macro_tickers['Silver'], level='Ticker')
                        #df_asset = fe_local.add_context(sxy_data, prefix='Silver')
                    except: pass
                # 4. Volatilité Pétrole (Pour Pétrole uniquement)
                if selected_ticker_hyb == 'USO':
                    try:
                        ovx_data = macro_data.xs(macro_tickers['OilVol'], level='Ticker')
                        df_asset = fe_local.add_context(ovx_data, prefix='OilVol')
                    except: pass

                # C. Préparation (Log Returns)
                prep = ModelPreparator(df_asset)
                prep.prepare_data()

                # D. Scaling Global x100 (Comme dans le notebook)
                # Cela permet de travailler avec des pourcentages (ex: 1.5 pour 1.5%)
                prep.data['Log Returns'] = prep.data['Log Returns'] * 100.0
                prep.data['Target'] = prep.data['Target'] * 100.0
                # On scale aussi les Lags
                for col in [c for c in prep.data.columns if 'Ret_Lag' in c]:
                    prep.data[col] = prep.data[col] * 100.0

                # F. Split Temporel
                # train_df, test_df = prep.train_test_split(train_size=train_size_hyb)
                train_df, test_df = prep.train_test_split_by_date(split_date_str=split_date_input)

                # E. Filtrage Lookback si nécessaire
                
                if lookback_option != "Tout l'historique":
                    # 1. Récupération du nombre d'années
                    years_back = int(lookback_option.split()[0]) # "3 Ans" -> 3
                    
                    # 2. Date de fin du Train (le "présent" au moment de l'entraînement)
                    split_date = train_df.index.get_level_values('Date').max()
                    
                    # 3. Calcul de la date de début acceptée
                    start_cutoff = split_date - pd.DateOffset(years=years_back)
                    
                    # 4. Filtrage
                    # On ne garde que les données entre (Split - X années) et (Split)
                    mask_lookback = train_df.index.get_level_values('Date') >= start_cutoff
                    train_df = train_df.loc[mask_lookback]
                    
                    # Sécurité
                    if len(train_df) < 20:
                        st.warning(f"⚠️ Attention : Période trop courte ({len(train_df)} points). Le modèle risque d'échouer.")
                
                st.write(f"📊 Données : **{len(train_df)}** semaines d'entraînement ({lookback_option}) | **{len(test_df)}** semaines de test")

                # =========================================================
                # 2. ARIMA (TENDANCE)
                # =========================================================
                status_container.info("Phase 2/4 : Entraînement ARIMA (Tendance)...")
                
                # On passe les rendements décimaux (/100) car StatisticalModeler refait x100 en interne
                y_train_series = train_df['Log Returns'].dropna()
                stat_modeler = StatisticalModeler(y_train_series / 100.0)
                
                # Fit ARIMA
                arima_model = stat_modeler.fit_arima()
                if arima_model is None:
                    st.error("ARIMA n'a pas convergé.")
                    st.stop()
                
                # arima_model.order retourne un tuple (p, d, q)
                p, d, q = arima_model.order
                st.success(f"✅ Tendance modélisée par **ARIMA({p}, {d}, {q})**")
                    
                # Prédiction ARIMA sur le Test (Baseline)
                arima_pred_test = arima_model.predict(n_periods=len(test_df))
                if not isinstance(arima_pred_test, pd.Series):
                    arima_pred_test = pd.Series(arima_pred_test, index=test_df.index)

                # Récupération des résidus sur le Train (Target pour XGBoost)
                resid_train_values = stat_modeler.residuals
                # Alignement taille (ARIMA perd parfois des points au début à cause de la différenciation)
                min_len = min(len(y_train_series), len(resid_train_values))
                resid_train = pd.Series(resid_train_values[-min_len:], index=y_train_series.index[-min_len:])

                # =========================================================
                # 3. XGBOOST (CORRECTION DES RÉSIDUS)
                # =========================================================
                status_container.info("Phase 3/4 : Entraînement XGBoost (Correction)...")

                # --- MODIFICATION MAJEURE : CIBLE DÉCALÉE (SHIFT) ---
                # On veut que les features à 't' prédisent le résidu à 't+1'
                # resid_train contient les résidus alignés temporellement
                
                # 1. Création de la Target "Résidu Futur"
                # On décale les résidus vers le haut (-1) : à la date t, on a le résidu de t+1
                y_target_resid = resid_train.shift(-1).dropna()

                # 2. Alignement des Index (Intersection Features X_t et Target Y_{t+1})
                # On ne garde que les dates présentes dans les deux (Features et Target Future)
                common_index_train = train_df.index.intersection(y_target_resid.index)
                
                if len(common_index_train) < 10:
                    st.error("Pas assez de données communes après le décalage (Shift) pour XGBoost.")
                    st.stop()

                # 3. Préparation des Features sur l'index aligné
                train_subset = train_df.loc[common_index_train]
                
                # On utilise get_X_y pour obtenir X_train transformé (Scale) et X_test
                # Note: X_test n'est pas affecté par le shift du train, c'est normal
                X_train_aligned, _, X_test, _, scaler = prep.get_X_y(train_subset, test_df)
                
                # 4. Sélection de la target alignée
                y_train_xgb = y_target_resid.loc[common_index_train]

                # Entraînement
                xgb_modeler = XGBoostModeler(n_estimators=n_estim, max_depth=2, learning_rate=0.05)
                xgb_modeler.train(X_train_aligned, y_train_xgb, use_custom_loss=False) # MSE pour la stabilité

                # Prédiction des résidus sur le Test
                xgb_pred_resid = xgb_modeler.predict(X_test)
                xgb_pred_resid = pd.Series(xgb_pred_resid, index=test_df.index)

                # =========================================================
                # 4. RÉSULTATS & RECONSTRUCTION
                # =========================================================
                status_container.success("Calculs terminés ! Génération des graphiques...")

                # Combinaison : Tendance + Correction
                final_pred_pct = arima_pred_test + xgb_pred_resid
                final_pred_decimal = final_pred_pct / 100.0 # Retour en décimales

                # =========================================================
                # NOUVEAU : MÉTRIQUES & GRAPHIQUE DES RENDEMENTS
                # =========================================================
                st.markdown("---")
                st.subheader("1. Analyse de la Performance (Rendements)")
                
                # 1. Calcul des Métriques sur les Log Returns (Échelle %)
                # Note : test_df['Log Returns'] est déjà x100 grâce à votre préparation
                actual_ret_pct = test_df['Log Returns']
                
                # RMSE & MAE sur la prédiction FINALE (Hybride)
                rmse_ret = np.sqrt(mean_squared_error(actual_ret_pct, final_pred_pct))
                mae_ret = mean_absolute_error(actual_ret_pct, final_pred_pct)
                dir_acc_xgb = np.mean(np.sign(actual_ret_pct) == np.sign(final_pred_pct))
                dir_acc_arima = np.mean(np.sign(actual_ret_pct) == np.sign(arima_pred_test))
                
                # Affichage des métriques
                col_met1, col_met2, col_met3 = st.columns(3)
                col_met1.metric("RMSE (Rendements)", f"{rmse_ret:.4f}%", help="Erreur quadratique moyenne sur les rendements hebdo.")
                col_met2.metric("MAE (Rendements)", f"{mae_ret:.4f}%", help="Erreur absolue moyenne (écart type moyen).")
                
                # Gain par rapport à ARIMA seul (Bonus info)
                rmse_arima = np.sqrt(mean_squared_error(actual_ret_pct, arima_pred_test))
                gain_vs_arima = (1 - rmse_ret/rmse_arima) * 100
                col_met3.metric("Amélioration vs ARIMA", f"{gain_vs_arima:.1f}%", delta="Impact XGBoost")

                # Précision directionnelle
                col_kpi1, col_kpi2 = st.columns(2)
                col_kpi1.metric("Précision Directionnelle (XGBoost)", f"{dir_acc_xgb:.2%}", delta=f"{(dir_acc_xgb-0.5)*100:.1f} pts vs Hasard")
                col_kpi2.metric("Précision Directionnelle (ARIMA)", f"{dir_acc_arima:.2%}", delta=f"{(dir_acc_arima-0.5)*100:.1f} pts vs Hasard")

                # 2. Graphique : Série Temporelle des Rendements (Train + Test + Pred)
                fig_ret = go.Figure()

                # A. Historique (Train) - En gris clair pour le contexte
                # On utilise get_level_values('Date') pour gérer le MultiIndex
                fig_ret.add_trace(go.Scatter(
                    x=train_df.index.get_level_values('Date'),
                    y=train_df['Log Returns'],
                    name='History (Train)',
                    line=dict(color='black', width=1.5),
                    hoverinfo='skip' # On allège le hover sur le passé lointain
                ))

                # B. Réalité (Test) - En gris foncé
                fig_ret.add_trace(go.Scatter(
                    x=test_df.index.get_level_values('Date'),
                    y=test_df['Log Returns'],
                    name='Actual (Test)',
                    line=dict(color='white', width=1.5, dash='dot')
                ))

                # C. Prédiction ARIMA (Baseline) - En Rouge
                # Cela permet de voir ce que le modèle linéaire "pensait" avant correction
                fig_ret.add_trace(go.Scatter(
                    x=test_df.index.get_level_values('Date'),
                    y=arima_pred_test,
                    name='ARIMA Prediction (Mean)',
                    line=dict(color="#FF0000", width=2)
                ))
                
                # Optionnel : Ajouter la prédiction Hybride finale en pointillé bleu pour comparer
                fig_ret.add_trace(go.Scatter(
                    x=test_df.index.get_level_values('Date'),
                    y=final_pred_pct,
                    name='Hybrid Prediction (Final)',
                    line=dict(color="#0818BE", width=2),
                    visible='legendonly' # Masqué par défaut pour ne pas surcharger, cliquable
                ))

                fig_ret.update_layout(
                    title=f"Weekly Log Returns: {selected_ticker_hyb}",
                    yaxis_title="Return (%)",
                    xaxis_title="Date",
                    height=450,
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(make_transparent(fig_ret), use_container_width=True)

                st.markdown("---")

                # CORRECTION DU PRIX RÉEL (Fix du bug d'échelle)
                # On divise temporairement Target par 100 pour que reconstruct_prices fonctionne
                prep.data['Target'] = prep.data['Target'] / 100.0
                results = prep.reconstruct_prices(predictions=final_pred_decimal, index=test_df.index)
                prep.data['Target'] = prep.data['Target'] * 100.0 # Rollback propre
                
                results = results.dropna()

                # =========================================================
                # RECONSTRUCTION PRIX : ARIMA SEUL (Pour comparaison)
                # =========================================================
                # On doit diviser par 100 car arima_pred_test est en pourcentage
                arima_pred_decimal = arima_pred_test / 100.0
                
                # On utilise la même fonction de reconstruction
                # Note: On triche un peu sur 'Target' pour que la fonction ne plante pas, 
                # mais on ne s'intéresse qu'à la colonne prédite.
                prep.data['Target'] = prep.data['Target'] / 100.0
                results_arima = prep.reconstruct_prices(predictions=arima_pred_decimal, index=test_df.index)
                prep.data['Target'] = prep.data['Target'] * 100.0 # Rollback
                results_arima = results_arima.dropna()

                # =========================================================
                # GRAPHIQUE PRIX : HISTORIQUE + TEST + MODÈLES
                # =========================================================
                st.subheader(f"Price Trajectory: {selected_ticker_hyb}")
                
                fig_price = go.Figure()

                # 1. HISTORIQUE (Train) - Pour le contexte
                # On prend les données de train_df pour afficher le passé
                fig_price.add_trace(go.Scatter(
                    x=train_df.index.get_level_values('Date'), 
                    y=train_df['Close'], 
                    name='History (Train)', 
                    line=dict(color='black', width=1),
                    hoverinfo='skip' # On allège le tooltip sur le passé
                ))

                # 2. PRIX RÉEL (Test)
                fig_price.add_trace(go.Scatter(
                    x=results.index.get_level_values('Date'), 
                    y=results['Actual_Close_t+1'], 
                    name='Actual Price (Test)', 
                    line=dict(color='black', width=2)
                ))

                # 3. PRÉDICTION ARIMA SEUL (Baseline)
                fig_price.add_trace(go.Scatter(
                    x=results_arima.index.get_level_values('Date'), 
                    y=results_arima['Predicted_Close_t+1'], 
                    name='ARIMA Alone (Mean)', 
                    line=dict(color='#FF9800', width=2, dash='dot')
                ))

                # 4. PRÉDICTION HYBRIDE (ARIMA + XGBoost)
                fig_price.add_trace(go.Scatter(
                    x=results.index.get_level_values('Date'), 
                    y=results['Predicted_Close_t+1'], 
                    name='Hybrid Model (Final)', 
                    line=dict(color='#2962FF', width=2, dash='dash')
                ))
                
                fig_price.update_layout(
                    title=f"Price Projection: Actual vs Models",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=550, # Un peu plus grand pour bien voir
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(make_transparent(fig_price), use_container_width=True)

                # =========================================================
                # GRAPHIQUE 2 : TRAJECTOIRE PURE (CUMULATIVE / RÉCURSIVE)
                # =========================================================
                st.markdown("---")
                st.subheader(f"Pure Trajectory (Without Recalibration)")
                st.caption("""
                This chart simulates what would have happened if you bought the asset at the start of the Test period 
                and let the model run autonomously, **without ever correcting the price with reality**. 
                It shows the accumulation of errors (drift) over the long term.
                """)

                # 1. Point de Départ : Le dernier prix connu du Train
                last_train_price = train_df['Close'].iloc[-1]

                # 2. Calcul de la trajectoire ARIMA (Cumulative)
                # Formule : P_t = P_0 * exp(somme_cumulée(r_t))
                arima_cum_returns = np.cumsum(arima_pred_decimal)
                arima_path_values = last_train_price * np.exp(arima_cum_returns)
                # Création Série Pandas
                arima_path = pd.Series(arima_path_values, index=test_df.index)

                # 3. Calcul de la trajectoire HYBRIDE (Cumulative)
                final_cum_returns = np.cumsum(final_pred_decimal)
                hybrid_path_values = last_train_price * np.exp(final_cum_returns)
                # Création Série Pandas
                hybrid_path = pd.Series(hybrid_path_values, index=test_df.index)

                # 4. Construction du Graphique
                fig_recursive = go.Figure()

                # A. Historique (Train) - Contexte
                fig_recursive.add_trace(go.Scatter(
                    x=train_df.index.get_level_values('Date'), 
                    y=train_df['Close'], 
                    name='History (Train)', 
                    line=dict(color='black', width=1.5),
                    hoverinfo='skip'
                ))

                # B. Prix Réel (Test)
                # On utilise test_df['Close'] directement qui contient les vrais prix
                fig_recursive.add_trace(go.Scatter(
                    x=test_df.index.get_level_values('Date'), 
                    y=test_df['Close'], 
                    name='Actual Price (Test)', 
                    line=dict(color='white', width=2, dash = 'dot')
                ))

                # C. Trajectoire ARIMA Pure
                fig_recursive.add_trace(go.Scatter(
                    x=arima_path.index.get_level_values('Date'), 
                    y=arima_path, 
                    name='ARIMA Trajectory (Drift)', 
                    line=dict(color="#FF0000", width=2, dash='dot')
                ))

                # D. Trajectoire Hybride Pure
                fig_recursive.add_trace(go.Scatter(
                    x=hybrid_path.index.get_level_values('Date'), 
                    y=hybrid_path, 
                    name='Hybrid Trajectory (Drift)', 
                    line=dict(color='#2962FF', width=2)
                ))

                fig_recursive.update_layout(
                    title=f"Long-Term Simulation: {selected_ticker_hyb}",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=550,
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Application de la transparence si vous avez ajouté la fonction helper
                if 'make_transparent' in globals():
                    fig_recursive = make_transparent(fig_recursive)
                
                st.plotly_chart(fig_recursive, use_container_width=True)

                # Metric sur Prix
                rmse_ret_price = np.sqrt(mean_squared_error(test_df['Close'], hybrid_path))
                mae_ret_price = mean_absolute_error(test_df['Close'], hybrid_path)
                
                # Affichage des métriques
                col_met1_price, col_met2_price = st.columns(2)
                col_met1_price.metric("RMSE (Price)", f"{rmse_ret_price:.2f}", help="Root Mean Squared Error on weekly prices.")
                col_met2_price.metric("MAE (Price)", f"{mae_ret_price:.2f}", help="Mean Absolute Error.")

                # --- GRAPHIQUE 2 : IMPORTANCE DES FEATURES ---
                st.subheader("Corrective Factors Analysis")
                st.caption("Which indicators does XGBoost use to correct ARIMA?")
                
                feats_names = prep.get_feature_names(train_subset)
                imp_df = xgb_modeler.get_feature_importance(feats_names)
                
                if imp_df is not None:
                    # On affiche les Top 10
                    top_imp = imp_df.tail(10)
                    fig_imp = go.Figure(go.Bar(
                        x=top_imp['Importance'],
                        y=top_imp['Feature'],
                        orientation='h',
                        marker=dict(color=top_imp['Importance'], colorscale='Reds')
                    ))
                    fig_imp.update_layout(height=400, title="Feature Importance (XGBoost)")
                    st.plotly_chart(make_transparent(fig_imp), use_container_width=True)

                # =========================================================
                # 5. ANALYSE WALK-FORWARD (SIMULATION RÉELLE AVEC GET_X_Y)
                # =========================================================
                st.markdown("---")
                st.subheader("2. Analyse Walk-Forward (Robustesse)")
                st.caption(f"Simulation pas-à-pas : Les modèles sont réentraînés chaque semaine sur la fenêtre '{lookback_option}' pour prédire la semaine suivante.")

                # Barre de progression
                wf_progress = st.progress(0)
                status_wf = st.empty()
                
                # 1. Configuration des données
                # On s'assure d'avoir tout l'historique trié
                full_data = prep.data.sort_index()
                test_indices = test_df.index 
                
                wf_preds = []
                wf_dates = []
                
                # Liste pour stocker les importances
                wf_importances_list = []
                
                # 2. Boucle Walk-Forward
                for i, idx in enumerate(test_indices):
                    # A. Mise à jour UI
                    prog = (i + 1) / len(test_indices)
                    wf_progress.progress(prog)
                    status_wf.text(f"Calcul semaine {i+1}/{len(test_indices)}...")
                    
                    # Récupération Date
                    current_date = pd.to_datetime(test_df.index.get_level_values('Date')[i])
                    
                    # B. Définition de la Fenêtre Glissante
                    if lookback_option == "Tout l'historique":
                        mask_train = full_data.index.get_level_values('Date') < current_date
                    else:
                        years = int(lookback_option.split()[0])
                        start_date_wf = current_date - pd.DateOffset(years=years)
                        mask_train = (full_data.index.get_level_values('Date') < current_date) & \
                                     (full_data.index.get_level_values('Date') >= start_date_wf)
                    
                    train_window = full_data.loc[mask_train]
                    
                    # Sécurité
                    if len(train_window) < 30:
                        wf_preds.append(0.0)
                        wf_dates.append(idx)
                        continue

                    try:
                        # --- C. MODÈLE 1 : ARIMA ---
                        y_train_wf = train_window['Log Returns'].dropna()
                        wf_stat_modeler = StatisticalModeler(y_train_wf / 100.0)
                        wf_stat_modeler.fit_arima()
                        
                        if wf_stat_modeler.arima_model:
                            pred_arima_val = wf_stat_modeler.arima_model.predict(n_periods=1)[0] 
                            
                            # Résidus
                            resid_values = wf_stat_modeler.residuals
                            min_len = min(len(y_train_wf), len(resid_values))
                            resid_train_series = pd.Series(resid_values[-min_len:], index=y_train_wf.index[-min_len:])
                        else:
                            pred_arima_val = 0.0
                            resid_train_series = pd.Series(0.0, index=y_train_wf.index)

                        # --- D. MODÈLE 2 : XGBOOST (CORRECTION) ---
                        y_xgb_target = resid_train_series.shift(-1).dropna()
                        common_idx = train_window.index.intersection(y_xgb_target.index)
                        
                        if len(common_idx) > 10:
                            # 1. & 2. Définition Train/Test
                            train_subset_wf = train_window.loc[common_idx]
                            test_subset_wf = full_data.loc[[idx]]
                            
                            # 3. get_X_y (Nettoyage + Scaling)
                            X_train_wf, _, X_test_step, _, _ = prep.get_X_y(train_subset_wf, test_subset_wf)
                            
                            # 4. Target Alignée
                            y_train_xgb_wf = y_xgb_target.loc[common_idx]
                            
                            # 5. Entraînement
                            wf_xgb_modeler = XGBoostModeler(n_estimators=n_estim, max_depth=2, learning_rate=0.05)
                            wf_xgb_modeler.train(X_train_wf, y_train_xgb_wf, use_custom_loss=False)
                            
                            # 6. Prédiction
                            pred_xgb_val = wf_xgb_modeler.predict(X_test_step)[0]
                            
                            # --- CORRECTION FEATURE IMPORTANCE ---
                            try:
                                # On utilise la méthode de la classe pour être sûr d'avoir les mêmes colonnes que get_X_y
                                feats_wf = prep.get_feature_names(train_subset_wf)
                                
                                imp_wf = wf_xgb_modeler.get_feature_importance(feats_wf)
                                if imp_wf is not None:
                                    wf_importances_list.append(imp_wf.set_index('Feature')['Importance'])
                            except: pass

                        else:
                            pred_xgb_val = 0.0
                        
                        # --- E. AGGRÉGATION ---
                        total_pred = pred_arima_val + pred_xgb_val
                        wf_preds.append(total_pred)
                        wf_dates.append(idx)
                        
                    except Exception as e:
                        wf_preds.append(0.0)
                        wf_dates.append(idx)

                status_wf.empty()
                
                # 3. Création des Résultats Walk-Forward
                pred_wf = pd.Series(wf_preds, index=test_df.index)
                actual_wf = test_df['Log Returns']
                
                pred_wf = pred_wf.astype(float)
                actual_wf = actual_wf.astype(float)
                
                rmse_wf = np.sqrt(mean_squared_error(actual_wf, pred_wf))
                dir_acc_wf = np.mean(np.sign(actual_wf) == np.sign(pred_wf))

                # 4. Affichage Graphique
                col_wf1, col_wf2 = st.columns(2)
                col_wf1.metric("RMSE (Walk-Forward)", f"{rmse_wf:.4f}%", delta=f"{(rmse_ret - rmse_wf):.4f} vs Static")
                col_wf2.metric("Précision Directionnelle", f"{dir_acc_wf:.2%}")

                fig_wf = go.Figure()
                fig_wf.add_trace(go.Scatter(
                    x=train_df.index.get_level_values('Date'),
                    y=train_df['Log Returns'],
                    name='History (Train)',
                    line=dict(color='black', width=1.5),
                    hoverinfo='skip'
                ))
                fig_wf.add_trace(go.Scatter(
                    x=test_df.index.get_level_values('Date'),
                    y=test_df['Log Returns'],
                    name='Actual (Test)',
                    line=dict(color='white', width=1.5, dash='dot')
                ))
                fig_wf.add_trace(go.Scatter(
                    x=pred_wf.index.get_level_values('Date'),
                    y=pred_wf,
                    name='Walk-Forward Prediction',
                    line=dict(color="#00C853", width=2)
                ))
                fig_wf.update_layout(
                    title=f"Walk-Forward Analysis: {selected_ticker_hyb} (Rolling Window: {lookback_option})",
                    yaxis_title="Return (%)", xaxis_title="Date", height=450,
                    hovermode="x unified", template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                if 'make_transparent' in globals():
                    st.plotly_chart(make_transparent(fig_wf), use_container_width=True)
                else:
                    st.plotly_chart(fig_wf, use_container_width=True)

                # =========================================================
                # 6. GRAPH DE L'IMPORTANCE MOYENNE (NOUVEAU)
                # =========================================================
                st.markdown("---")
                st.subheader("📊 Average Feature Importance (Walk-Forward)")
                st.caption(f"Importance moyenne des indicateurs calculée sur {len(wf_importances_list)} ré-entraînements successifs.")
                
                if wf_importances_list:
                    # 1. Concaténation de toutes les séries
                    df_all_imps = pd.concat(wf_importances_list, axis=1).fillna(0)
                    
                    # 2. Calcul de la moyenne par feature
                    avg_imp_series = df_all_imps.mean(axis=1).sort_values(ascending=True)
                    
                    # 3. Préparation pour Plotly (Top 15)
                    top_avg_imp = avg_imp_series.tail(15)
                    
                    fig_imp_wf = go.Figure(go.Bar(
                        x=top_avg_imp.values,
                        y=top_avg_imp.index,
                        orientation='h',
                        marker=dict(color=top_avg_imp.values, colorscale='Reds')
                    ))
                    
                    fig_imp_wf.update_layout(
                        title="Quels indicateurs ont été les plus utiles sur la durée ?",
                        xaxis_title="Importance Moyenne (Gain)",
                        height=500,
                        template="plotly_white"
                    )
                    
                    if 'make_transparent' in globals():
                        st.plotly_chart(make_transparent(fig_imp_wf), use_container_width=True)
                    else:
                        st.plotly_chart(fig_imp_wf, use_container_width=True)
                else:
                    st.warning("Pas assez de données pour calculer l'importance moyenne.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                # Affiche la trace pour le debug si besoin
                st.exception(e)

# --- PAGE 3 : BACKTEST ---

elif selection == "3. Backtesting & Performance":
    st.title("⚖️ Allocation Stratégique (Markowitz Actif)")
    st.markdown("""
    **Stratégie Hebdomadaire (Smart Beta Long/Short) :**
    1.  **Filtre de Tendance** : Achat si Haussier, Vente si Baissier (Bornes dynamiques).
    2.  **Rendement Espéré** : Modèle Hybride (ARIMA + XGBoost sur résidus).
    3.  **Gestion des Régimes** : Choix entre filtre binaire (SMA) ou probabiliste (HMM)
    4.  **Risque (Covariance)** : Volatilité GARCH prédite + Corrélation Historique.
    5.  **Optimisation** : Maximisation du Ratio de Sharpe.
    """)
    st.title("📝 Validation & Backtesting")
    st.markdown("""
    **Protocole Walk-Forward (Mensuel) :**
    * Simulation pas-à-pas avec "Buffer" de stabilité pour limiter le turnover.
    * Prise en compte stricte des frais et du drift.
    """)
    
    from src.backtest import WalkForwardBacktest
    
    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        years_back = st.selectbox("Durée du Backtest", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15], index=4, format_func=lambda x: f"{x} Ans")
        start_date_bt = pd.Timestamp.now() - pd.Timedelta(days=18) - pd.DateOffset(years=years_back)
        
    with col_bt2:
        capital_bt = st.number_input("Capital Initial ($)", value=10000)
    
    with col_bt3:
        # Ajout de l'option Tranching
        use_tranching = st.checkbox("Activer le Tranching (Comparatif)", value=True, 
                                  help="Lance 4 simulations décalées pour comparer la robustesse des timings.")
        
    # --- NOUVEAU SÉLECTEUR DE RÉGIME ---
    st.divider()
    st.subheader("⚙️ Paramètres de Gestion du Risque")
    
    regime_choice = st.radio(
        "Méthode de Détection de Régime & Allocation :",
        ["🛡️ Filtre de Tendance Classique (SMA 50/200)", "🤖 Modèle de Markov Caché (HMM + Probabilités)"],
        index=0,
        help="SMA : Coupe les positions si tendance baissière (Binaire). HMM : Réduit la taille des positions proportionnellement à la probabilité de Krach (Nuancé)."
    )
    
    # Mapping vers le code (sma ou hmm)
    regime_method_code = "hmm" if "Markov" in regime_choice else "sma"
    
    if regime_method_code == "hmm":
        st.info("ℹ️ **Mode HMM Activé** : L'algorithme calculera la probabilité de régime Baissier (Bear Prob) à chaque rebalancement et ajustera l'exposition (Cash Buffer) en conséquence.")
    else:
        st.info("ℹ️ **Mode SMA Activé** : L'algorithme interdira l'achat (Position = 0) sur les actifs en tendance baissière (Prix < SMA50 < SMA200).")

 
    # --- GESTION DE L'ÉTAT (SESSION STATE) ---
    if st.button("Lancer le Backtest Mensuel", type="primary"):
        with st.spinner("Simulation en cours..."):
            try:
                # Initialisation du moteur
                bt_engine = WalkForwardBacktest(
                    full_data=df, 
                    tickers=USER_PORTFOLIO, 
                    benchmark_ticker='^GSPC',
                    initial_capital=capital_bt
                )
                
                if use_tranching:
                    # --- MODE 1 : TRANCHING (Multi-Courbes) ---
                    # Retourne un DICTIONNAIRE de résultats : {'Tranche_0': {...}, 'Tranche_1': ...}
                    tranches_results = bt_engine.run_with_tranching(
                        start_date=start_date_bt.strftime('%Y-%m-%d'),
                        step_weeks=4,
                        turnover_buffer=0.40,
                        regime_method=regime_method_code
                    )
                    st.session_state['bt_tranches_results'] = tranches_results
                    st.session_state['bt_mode'] = 'tranching'
                    st.info("ℹ️ Mode Tranching : 4 scénarios générés pour analyse comparative.")
                    
                else:
                    # --- MODE 2 : STANDARD (Unique) ---
                    # Retourne un DATAFRAME unique
                    results_df = bt_engine.run(
                        start_date=start_date_bt.strftime('%Y-%m-%d'),
                        step_weeks=4,
                        turnover_buffer=0.40,
                        regime_method=regime_method_code
                    )
                    metrics = bt_engine.get_metrics(results_df)
                    
                    st.session_state['bt_results'] = results_df
                    st.session_state['bt_metrics'] = metrics
                    st.session_state['bt_mode'] = 'standard'
                    st.info("⚠️ Mode Standard (Sensible au Timing).")
                
                # Sauvegarde du moteur pour les plots
                st.session_state['bt_engine'] = bt_engine 
                st.session_state['bt_run_done'] = True
                
                st.success("Backtest Terminé avec succès !")
                
            except Exception as e:
                st.error(f"Erreur durant le backtest : {e}")
                st.exception(e)

    # --- AFFICHAGE DES RÉSULTATS ---
    if st.session_state.get('bt_run_done', False):
        
        bt_engine = st.session_state['bt_engine']
        mode = st.session_state.get('bt_mode', 'standard')
        
        st.divider()

        # ==============================================================================
        # CAS 1 : AFFICHAGE MODE TRANCHING (COMPARAISON ROBUSTESSE)
        # ==============================================================================
        if mode == 'tranching':
            results = st.session_state['bt_tranches_results']
            
            # --- 1. CALCUL DES MÉTRIQUES MOYENNES (TOUTES LES MÉTRIQUES) ---
            # On stocke toutes les métriques de chaque tranche dans une liste
            all_metrics_dicts = []
            for t_name, data in results.items():
                m = bt_engine.get_metrics(data['df'])
                all_metrics_dicts.append(m)
            
            # On crée un DataFrame pour faciliter la moyenne
            df_metrics = pd.DataFrame(all_metrics_dicts)
            avg_metrics = df_metrics.mean() # Moyenne colonne par colonne
            
            # Métriques de dispersion (Pour voir si les tranches sont très différentes)
            # Sharpe Min/Max
            min_sharpe = df_metrics['Sharpe Ratio'].min()
            max_sharpe = df_metrics['Sharpe Ratio'].max()
            
            st.subheader(f"📊 Synthèse Moyenne ({len(results)} Stratégies Décalées)")
            
            # Ligne 1 : Performance & Risque Principal
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR Moyen", f"{avg_metrics['CAGR (Annuel)']:.2%}")
            k2.metric("Sharpe Moyen", f"{avg_metrics['Sharpe Ratio']:.2f}", help=f"Min: {min_sharpe:.2f} | Max: {max_sharpe:.2f}")
            k3.metric("Max Drawdown Moyen", f"{avg_metrics['Max Drawdown']:.2%}")
            k4.metric("Win Rate Moyen", f"{avg_metrics['Win Rate']:.2%}")
            
            # Ligne 2 : Métriques Avancées (Alpha, Beta, Volatilité, VaR)
            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Alpha Moyen", f"{avg_metrics['Alpha']:.2%}", help="Surperformance moyenne vs Benchmark")
            k6.metric("Beta Moyen", f"{avg_metrics['Beta']:.2f}", help="Sensibilité moyenne au marché")
            k7.metric("Volatilité Moyenne", f"{avg_metrics['Volatilité']:.2%}")
            # On gère le cas où VaR n'existe pas (si backtest trop court)
            var_val = avg_metrics.get('VaR (95%)', 0.0)
            k8.metric("VaR (95%) Moyenne", f"{var_val:.2%}")
            
            # Ligne 3 : Risque de Queue (Kurtosis, Skewness)
            k9, k10, k11, k12 = st.columns(4)
            k9.metric("Kurtosis Moyen", f"{avg_metrics.get('Kurtosis', 0.0):.2f}")
            k10.metric("Skewness Moyen", f"{avg_metrics.get('Skewness', 0.0):.2f}")
            k11.metric("Expected Shortfall (CVaR)", f"{avg_metrics.get('CVaR (95%)', 0.0):.2f}")
            k12.metric("Dispersion Sharpe", f"{(max_sharpe - min_sharpe):.2f}", help="Écart entre le meilleur et le pire timing.")

            st.divider()

            # --- ONGLETS DE VISUALISATION ---
            # Ajout de l'onglet "📅 Analyse Annuelle"
            tab_comp, tab_annual, tab_alloc_comp, tab_frais_comp = st.tabs([
                "📈 Performance Globale", 
                "📅 Analyse Annuelle", 
                "📊 Allocations", 
                "💸 Frais"
            ])
            
            # --- Onglet 1 : Performance Cumulée ---
            with tab_comp:
                st.markdown("#### Comparaison aux Benchmarks et entre Tranches")
                fig_equity = bt_engine.plot_tranching_results(results)
                st.plotly_chart(make_transparent(fig_equity), use_container_width=True)

            # --- Onglet 2 : Analyse Annuelle (AVEC SOUS-ONGLETS) ---
            with tab_annual:
                st.markdown("#### Détail par Année Civile")
                st.caption("Comparez la capacité de chaque variante de la stratégie à battre le marché année après année.")
                
                # Récupération des 2 figures
                fig_returns, fig_drawdowns = bt_engine.plot_annual_stats_comparison(results)
                
                # CRÉATION DES DEUX SOUS-ONGLETS
                subtab_ret, subtab_dd = st.tabs(["💰 Rendements Annuels", "📉 Drawdowns Annuels"])
                
                with subtab_ret:
                    st.plotly_chart(make_transparent(fig_returns), use_container_width=True)
                    st.info("💡 **Lecture :** Une année est réussie si les barres colorées (Stratégies) sont plus hautes que la barre noire (Benchmark).")
                
                with subtab_dd:
                    st.plotly_chart(make_transparent(fig_drawdowns), use_container_width=True)
                    st.error("💡 **Lecture :** Plus la barre descend bas, plus le risque subi cette année-là a été violent. On cherche des barres colorées plus courtes que la noire.")

            # --- Onglet 3 : Allocations ---
            with tab_alloc_comp:
                st.markdown("#### Détail des positions : Décalage temporel")
                fig_alloc = bt_engine.plot_tranching_allocations(results)
                st.plotly_chart(make_transparent(fig_alloc), use_container_width=True)

            # --- Onglet 4 : Frais ---
            with tab_frais_comp:
                st.markdown("#### Impact du Timing sur le Turnover")
                fig_costs = bt_engine.plot_tranching_costs(results)
                st.plotly_chart(make_transparent(fig_costs), use_container_width=True)

        # ==============================================================================
        # CAS 2 : AFFICHAGE MODE STANDARD (UNIQUE)
        # ==============================================================================
        else:
            # Récupération depuis la session
            metrics = st.session_state['bt_metrics']
            results_df = st.session_state['bt_results']
            
            # --- KPI ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("CAGR (Annuel)", f"{metrics['CAGR (Annuel)']:.2%}")
            kpi2.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            kpi3.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
            kpi4.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
            
            kpi5, kpi6, kpi7, kpi8 = st.columns(4)
            kpi5.metric("Alpha", f"{metrics['Alpha']:.2%}")
            kpi6.metric("Beta", f"{metrics['Beta']:.2f}")
            kpi7.metric("Volatilité", f"{metrics['Volatilité']:.2%}")
            kpi8.metric("VaR (95%)", f"{metrics.get('VaR (95%)', 0.0):.2%}")
            
            # --- ONGLETS GRAPHIQUES ---
            tab_perf, tab_alloc, tab_frais, tab_macro, tab_data = st.tabs(["📈 Performance", "📊 Allocation", "💸 Frais", "🌍 Macro", "📄 Données"])
            
            with tab_perf:
                st.subheader("Courbe de Capital & Benchmarks")
                fig_bt = bt_engine.plot_results(results_df)
                st.plotly_chart(make_transparent(fig_bt), use_container_width=True)
            
            with tab_alloc:
                st.subheader("Dynamique du Portefeuille")
                fig_alloc = bt_engine.plot_allocation_evolution()
                if fig_alloc: st.plotly_chart(make_transparent(fig_alloc), use_container_width=True)

            with tab_frais:
                st.subheader("Analyse des Coûts")
                fig_costs = bt_engine.plot_cost_analysis(results_df)
                st.plotly_chart(make_transparent(fig_costs), use_container_width=True)
                
            with tab_macro:
                # Code existant pour la macro (inchangé)
                col_m1, col_m2 = st.columns([1, 3])
                with col_m1:
                    macro_options = {"VIX": "^VIX", "Taux 10A": "^TNX", "Dollar": "DX-Y.NYB"}
                    lbl = st.selectbox("Indicateur", list(macro_options.keys()))
                    mode_c = st.radio("Comparer", ["Capital", "Drawdown"])
                with col_m2:
                    fig_m, corr = bt_engine.plot_macro_correlation(results_df, df, macro_options[lbl], lbl, mode_c)
                    if fig_m: 
                        st.plotly_chart(make_transparent(fig_m), use_container_width=True)
                        st.info(f"Corrélation: {corr:.2f}")

            with tab_data:
                st.dataframe(results_df)

# --- PAGE 4 : LIVE TRADING DASHBOARD ---

if selection == "4. Live Trading Dashboard":
    st.title("⚡ Live Trading Assistant")
    st.markdown("""
    Ce module connecte votre stratégie à la réalité. 
    Il utilise **exactement les mêmes modèles** (ARIMA + XGBoost) et la même logique d'optimisation que le Backtest, 
    mais entraînés sur les données les plus récentes disponibles (jusqu'à la clôture d'hier).
    """)

    from src.live_dashboard import LiveDashboard
    
    # Instanciation
    live_engine = LiveDashboard(USER_PORTFOLIO)
    
    # --- PARTIE 1 : SAISIE DU PORTEFEUILLE ACTUEL ---
    st.subheader("1. Votre Portefeuille Actuel")
    
    c_input1, c_input2 = st.columns([2, 1])
    
    with c_input1:
        st.info("Entrez le nombre d'actions que vous possédez actuellement pour chaque actif.")
        
        # Initialisation du session state pour les positions si non existant
        if 'live_positions' not in st.session_state:
            # Création d'un DF par défaut
            init_data = {'Ticker': USER_PORTFOLIO, 'Quantité (Actions)': [0.0] * len(USER_PORTFOLIO)}
            st.session_state['live_positions'] = pd.DataFrame(init_data)
        
        # Éditeur de données
        edited_df = st.data_editor(
            st.session_state['live_positions'],
            column_config={
                "Quantité (Actions)": st.column_config.NumberColumn(
                    "Quantité",
                    help="Nombre de titres détenus (positif pour Long, négatif pour Short)",
                    min_value=-10000,
                    max_value=10000,
                    step=1,
                )
            },
            hide_index=True,
            use_container_width=True,
            key="editor_pos"
        )
        
        # Conversion en dictionnaire pour le traitement
        positions_dict = dict(zip(edited_df['Ticker'], edited_df['Quantité (Actions)']))

    with c_input2:
        st.write("### Cash Disponible")
        cash_input = st.number_input(
            "Liquidités ($)", 
            min_value=0.0, 
            value=10000.0, 
            step=100.0,
            help="Argent non investi disponible pour le trading."
        )
        
        lookback_live = st.selectbox(
            "Fenêtre d'Entraînement",
            ["1 An", "3 Ans", "5 Ans", "Tout l'historique"],
            index=1,
            help="Durée de l'historique utilisée pour ré-entraîner les modèles ARIMA/XGBoost."
        )
        years_map = {"1 An": 1, "3 Ans": 3, "5 Ans": 5, "Tout l'historique": 20}

    # Calcul en temps réel de la valeur estimée
    if st.button("🔄 Actualiser la valeur du portefeuille"):
        with st.spinner("Récupération des derniers prix..."):
            latest_prices = live_engine.get_latest_prices()
            weights_series, total_val = live_engine.calculate_current_portfolio(positions_dict, cash_input, latest_prices)
            
            st.metric("Valeur Totale Estimée (NAV)", f"{total_val:,.2f} $")
            st.write("**Répartition Actuelle estimée :**")
            
            # Petit graph bar pour voir où on en est
            df_curr = pd.DataFrame({'Actif': weights_series.index, 'Poids': weights_series.values})
            st.bar_chart(df_curr.set_index('Actif'))

    st.divider()

    # --- PARTIE 2 : LANCEMENT DE L'OPTIMISATION ---
    st.subheader("2. Génération des Ordres (IA Hybride)")
    
    if st.button("🚀 Lancer l'Analyse & Optimisation", type="primary"):
        
        # 1. Récupération prix frais
        with st.spinner("Initialisation..."):
            latest_prices = live_engine.get_latest_prices()
            current_weights_series, total_capital = live_engine.calculate_current_portfolio(positions_dict, cash_input, latest_prices)
        
        # 2. Exécution du Pipeline
        # On affiche les logs dans un expander pour voir ce qui se passe
        with st.status("Exécution de la stratégie en cours...", expanded=True) as status:
            
            df_orders, logs, regimes = live_engine.run_live_analysis(
                current_weights=current_weights_series,
                total_capital=total_capital,
                lookback_years=years_map[lookback_live]
            )
            
            for log in logs:
                st.write(log)
                
            status.update(label="Optimisation terminée !", state="complete", expanded=False)
            
        # 3. Affichage des Résultats
        st.success("✅ Allocation optimale calculée.")
        
        tab_res1, tab_res2 = st.tabs(["📋 Tableau des Ordres", "📊 Analyse Allocation"])
        
        with tab_res1:
            st.markdown("### 🛒 Ordres à exécuter pour la semaine prochaine")
            
            # Formatage pour affichage propre
            display_cols = ['Prix Actuel', 'Allocation Actuelle (%)', 'Allocation Cible (%)', 'Différence (%)', 'Ordre ($)', 'Ordre (Qté)']
            
            # Fonction de style pour colorer les ordres Achat/Vente
            def color_orders(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'grey'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                df_orders[display_cols].style.format({
                    'Prix Actuel': "{:.2f} $",
                    'Allocation Actuelle (%)': "{:.2f}%",
                    'Allocation Cible (%)': "{:.2f}%",
                    'Différence (%)': "{:+.2f}%",
                    'Ordre ($)': "{:+,.2f} $",
                    'Ordre (Qté)': "{:+.2f}"
                }).applymap(color_orders, subset=['Ordre ($)', 'Ordre (Qté)']),
                use_container_width=True
            )
            
            st.info("💡 **Interprétation :** 'Ordre (Qté)' indique combien d'actions vous devez acheter (+) ou vendre (-) pour atteindre l'allocation idéale calculée par l'IA.")

        with tab_res2:
            c_chart1, c_chart2 = st.columns(2)
            
            with c_chart1:
                st.markdown("#### Prédictions du Modèle")
                st.caption("Ce que le modèle anticipe pour la semaine prochaine.")
                st.dataframe(
                    df_orders[['Rendement Espéré (Semaine)', 'Volatilité (Semaine)']].style.format("{:.2f}%").background_gradient(cmap='Greens'),
                    use_container_width=True
                )
            
            with c_chart2:
                st.markdown("#### Régimes de Marché Détectés")
                st.caption("Filtre Safety-First (SMA 50/200)")
                for t, (mini, maxi) in regimes.items():
                    if maxi == 0:
                        status = "🔴 Baissier (Vente/Cash autorisé)"
                    elif mini == 0:
                        status = "🟢 Haussier (Achat autorisé)"
                    else:
                        status = "⚪ Neutre" # Cas rare selon config
                    
                    st.write(f"**{t}** : {status}")

            st.markdown("#### Comparaison Avant / Après")
            
            fig_alloc = go.Figure()
            fig_alloc.add_trace(go.Bar(
                x=df_orders.index,
                y=df_orders['Allocation Actuelle (%)'],
                name='Actuel',
                marker_color='grey'
            ))
            fig_alloc.add_trace(go.Bar(
                x=df_orders.index,
                y=df_orders['Allocation Cible (%)'],
                name='Cible (IA)',
                marker_color='#2962FF'
            ))
            fig_alloc.update_layout(title="Rééquilibrage du Portefeuille", barmode='group', template="plotly_white")
            st.plotly_chart(fig_alloc, use_container_width=True)