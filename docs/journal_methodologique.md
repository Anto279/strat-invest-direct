# Journal méthodologique

Corrections apportées lors de la refonte de la pipeline (version 2). Ce journal documente les biais identifiés dans la première version et la façon dont ils ont été traités. Les résultats publiés dans le README sont tous issus de la version corrigée.

| Problème identifié | Correction |
|---|---|
| Rendement du portefeuille calculé comme somme pondérée de **log-rendements** | Comptabilité en rendements simples avec dérive hebdomadaire des poids |
| Signal et exécution sur la **même clôture** (look-ahead) | Exécution à la clôture suivante (`execution_lag = 1`) |
| Barres `1wk` de Yahoo incluant la **semaine en cours incomplète** | Agrégation hebdomadaire depuis les données quotidiennes, semaine incomplète exclue |
| Benchmark `^GSPC` **hors dividendes** vs ETF en rendement total | Benchmark 60/40 en rendement total, mêmes coûts, plus référence 1/N |
| SMA 50/200 appliquées à des données **hebdomadaires** (SMA 200 = 4 ans) | SMA 30/40/52 semaines (équivalents 7, 10 et 12 mois), score multi-horizon |
| **Ventes à découvert** en régime baissier, incohérentes avec Safety-First | Long-only ; la poche non investie est placée en T-Bills |
| Max-Sharpe sous Σ\|w\| ≤ 1 **mal posé** (invariance d'échelle), r_f/52 pour un horizon de 4 semaines | Utilité moyenne-variance à l'horizon, plafond de volatilité, unités cohérentes |
| Taux sans risque **fixe à 2 %** | Série des T-Bills 13 semaines |
| Z-score glissant calculé sur des données concaténées, features **non stationnaires** | Features stationnaires, pas de normalisation (inutile pour les arbres) |
| XGBoost non régularisé hors échantillon | Calibration β par validation croisée purgée, régularisation renforcée |
| AIC via `auto_arima` sur-paramétrant | Grille exhaustive au BIC, AICc reporté pour contrôle |
| GARCH évalué contre sa **propre** volatilité ajustée ex-post (circularité) | QLIKE / MSE contre le proxy e², test DM contre EWMA, Mincer-Zarnowitz |
| Buffer de **40 %** de turnover, pouvant bloquer les sorties de risque | Buffer de 5 % one-way avec priorité absolue aux sorties de risque |
| Tranches affichées séparément, sans portefeuille agrégé | Agrégation des 4 tranches et publication de leur dispersion |
| Sharpe = (CAGR − 2 %)/σ, mélange géométrique / arithmétique | Moyenne arithmétique des rendements excédentaires × 52 / σ |
| Univers redondant (QQQ, LQD) | Une poche par prime de risque, validée par clustering |
| HMM instable | Supprimé |
