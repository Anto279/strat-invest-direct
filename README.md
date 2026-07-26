# 📈 Stratégie d'Investissement Directionnelle & Machine Learning

**Auteur** : Antonin Meudic  
**Type** : Stratégie quantitative directionnelle, hybride et multi-assets.

## 🎯 Genèse et Objectif du Projet

Ce projet naît d'une réflexion profonde sur la viabilité à long terme du système des retraites en France[cite: 11]. Face à ce constat, la nécessité d'investir dès aujourd'hui de manière autonome et sécurisée devient primordiale[cite: 11]. 

L'objectif principal est de concevoir une stratégie d'investissement automatisée, applicable par le grand public via un plan d'investissement mensuel programmé (DCA intelligent)[cite: 11]. Pour assurer la liquidité et minimiser les frottements, l'univers d'investissement est strictement composé d'actifs accessibles et à faibles frais, spécifiquement des ETF[cite: 11].

L'architecture repose sur une philosophie **"Safety-First"** : la préservation du capital (contrôle du Drawdown) prime sur la recherche absolue de rendement[cite: 11].

---

## 🧬 Univers d'Investissement (Global Macro)

Pour que l'optimisation mathématique soit viable, le portefeuille doit être composé d'actifs structurellement décorrélés (actions, obligations, matières premières, immobilier). Un univers purement "Actions US" rendrait la matrice de covariance mal conditionnée.

*   **Moteurs de Performance (Risk-On)** : S&P 500 (`SPY`), Nasdaq 100 (`QQQ`), Marchés Développés (`EFA`), Marchés Émergents (`EEM`).
*   **Amortisseurs (Risk-Off)** : Treasuries US 20+ ans (`TLT`), Treasuries US 7-10 ans (`IEF`), Corporate Bonds (`LQD`).
*   **Actifs Réels & Couverture** : Or Physique (`GLD`), Broad Commodities (`DBC`), Immobilier US (`VNQ`).

---

## 🧮 Architecture Mathématique et Statistique

La prise de décision s'articule autour de pipelines statistiques exécutés indépendamment sur **chaque actif**, afin d'établir des régimes de marché spécifiques et non globaux.

### 1. Modélisation Hybride des Rendements (ARIMA + XGBoost)
Prédire les marchés financiers revient à chercher un signal extrêmement faible dans un environnement ultra-bruité. La modélisation s'appuie sur une décomposition du rendement temporel : $R_t = f(t) + \epsilon_t$.

*   **ARIMA (Tendance Linéaire)** : Le modèle paramétrique capte l'autocorrélation simple et la tendance moyenne de l'actif[cite: 11].
*   **Ensemble XGBoost (Correction Non-Linéaire)** : Une forêt d'arbres de gradient boosting (intégrant du subsampling pour la robustesse) est entraînée spécifiquement pour prévoir les erreurs (les résidus $\epsilon_t$) du modèle ARIMA[cite: 11].
    *   *Features Techniques* : RSI, Bandes de Bollinger (Width et %B), MACD, et Lagged Returns (1, 2, 3, 5)[cite: 11].
    *   *Features Macroéconomiques* : VIX (Peur), Taux 10 ans, Dollar Index (DXY)[cite: 11].
    *   *Scaling* : Un Rolling Z-Score est appliqué pour éviter le *Data Leakage* et s'adapter aux changements de volatilité structurelle.

### 2. Modélisation du Risque (GARCH)
La volatilité financière n'est pas constante (hétéroscédasticité conditionnelle). Utiliser un écart-type historique simple sous-estime le risque lors des crises.
*   Le modèle **GARCH(1,1)** est estimé chaque mois sur chaque actif pour capturer le phénomène de *Volatility Clustering* (regroupement de volatilité)[cite: 11]. Les prévisions de variance sont ensuite réinjectées dans la matrice de covariance pour l'optimisation.

### 3. Le Filtre de Régime "Safety-First"
Le système n'est autorisé à prendre des positions longues que si l'environnement probabiliste est favorable[cite: 11]. Deux méthodes de détection de régime cohabitent :
*   **Filtre Tendance (Classique)** : Une approche binaire autorisant l'achat si $\text{Prix} > \text{SMA}_{50} > \text{SMA}_{200}$[cite: 11] et limitant à la vente à découvert (ou Cash) si $\text{Prix} < \text{SMA}_{50} < \text{SMA}_{200}$[cite: 11].
*   **Modèle de Markov Caché (HMM)** : Une approche non-supervisée probabiliste. Le HMM identifie les états cachés du marché via un algorithme d'Espérance-Maximisation (Baum-Welch). 
    *   *Justification Quant* : Contrairement à une approche naïve qui associerait l'état baissier à la variance maximale, notre HMM identifie le régime "Bear" comme l'état minimisant le Ratio de Sharpe $\arg\min(\mu / \sigma)$. Cela empêche le système de couper les actifs refuges (comme l'Or ou les Obligations) dont la volatilité explose *à la hausse* pendant les paniques boursières.

### 4. Allocation et Optimisation (Markowitz Actif Régularisé)
Une fois les espérances de rendements ($\mu$) et la matrice de covariance ($\Sigma$) calculées, l'allocation est résolue sous contraintes.
*   **Objectif** : Maximisation du Ratio de Sharpe net de frais[cite: 11].
*   **Règle "Cash is a Position"** : La somme des valeurs absolues des poids est contrainte à un maximum de 1 ($\sum |w_i| \leq 1$). Le résidu non alloué devient implicitement une poche de liquidité[cite: 11], offrant la possibilité de passer entièrement en cash si aucun actif n'est attractif[cite: 11].
*   **Pénalité L2 (Ridge)** : L'optimiseur classique de Markowitz étant un "maximiseur d'erreurs", une pénalité L2 ($\lambda \sum w_i^2$) est intégrée à la fonction objective. Cela force mathématiquement la diversification et empêche le modèle d'allouer 100% du capital au seul actif ayant la prédiction marginale la plus haute.
*   **Turnover & Frais** : Les coûts de transaction sont intégrés directement comme pénalité dans la fonction de minimisation de Scipy[cite: 11].

---

## 🧘‍♂️ Exécution et "Lazy Trading"

La sur-réaction au bruit du marché détruit l'Alpha à cause des frais de frottement. La stratégie intègre un mécanisme de seuil (Trade Buffer)[cite: 11].
*   Si la différence entre l'allocation cible et l'allocation actuelle entraîne un turnover inférieur à **10%** : l'ordre est ignoré[cite: 11].
*   Ce filtre passe-bas assure que l'algorithme ne déclenche un rebalancement que si le signal de marché est statistiquement significatif[cite: 11], réduisant drastiquement les frais de transaction et améliorant le Ratio de Sharpe Net[cite: 11].

---

## 🧪 Protocole de Validation (Backtest Walk-Forward)

Pour éviter l'overfitting, le modèle est validé via un algorithme de *Walk-Forward* strict.
1.  **Étanchéité temporelle** : Aucune donnée future n'est utilisée[cite: 11]. À chaque pas de temps $T$, le modèle est ré-entraîné uniquement sur la fenêtre glissante des 3 dernières années[cite: 11].
2.  **Exécution réaliste** : Calcul du turnover, soustraction des frais, et gestion du *drift* (dérive des poids due à l'évolution divergente des prix entre $T$ et $T+1$)[cite: 11].
3.  **Tranching** : Le backtest génère des simulations décalées temporellement pour valider la robustesse de la stratégie face à la sensibilité du timing (Day-of-the-month effect).

---

## 🛠️ Stack Technique

*   **Langage** : Python 3.11+
*   **Séries Temporelles & Économétrie** : `statsmodels`, `pmdarima`, `arch`
*   **Machine Learning** : `xgboost`, `scikit-learn`, `hmmlearn`
*   **Optimisation Mathématique** : `scipy.optimize`
*   **Data & Dataviz** : `pandas`, `numpy`, `plotly`, `streamlit`

---
*Ce projet démontre comment coupler l'économétrie classique aux algorithmes d'apprentissage automatique pour construire une solution de gestion de portefeuille de niveau institutionnel, orientée vers la protection du capital.*