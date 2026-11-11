# Fiche de lecture – Projet 9 : Deep Hedging

## 1. Contexte et objectif du projet

Une option financière est un contrat qui donne à son acheteur le droit d'acheter ou de vendre un actif à un prix fixé à l'avance (prix d'exercice \(K\)). Pour limiter les risques liés aux variations du marché, les institutions financières utilisent des stratégies de couverture (_Hedging_).

Notre projet consiste à comparer trois stratégies :

- **Sans couverture :** aucune position n'est prise sur le sous-jacent.
- **Couverture Delta (Black-Scholes) :** le nombre d'actions détenues est déterminé à partir d'une formule mathématique.
- **Deep Hedging :** un réseau de neurones apprend à déterminer les positions optimales afin de réduire le risque financier, notamment en présence de coûts de transaction.

## 2. Les principales formules financières

**A. Payoff d'une option Call**

\[
Z\_{\text{call}}=\max(S_T-K,0)
\]

Cette formule représente le gain à l'échéance : l'option rapporte la différence entre le prix du marché et le prix d'exercice lorsque celle-ci est positive.

**B. Delta de Black-Scholes**

\[
\Delta\_{\text{call}}=N(d_1)
\]

\[
d_1=\frac{\ln(S_0/K)+(r+\sigma^2/2)T}
{\sigma\sqrt{T}}
\]

Le Delta mesure la sensibilité du prix de l'option à une variation du prix du sous-jacent. Il indique la quantité d'actions nécessaire pour réaliser une couverture classique.

**C. Profit et perte (P&L)**

\[
P\&L=Z+\sum*{k=0}^{n-1}\delta_k(S*{k+1}-S*k)
-\sum*{k=0}^{n-1}c|\delta*k-\delta*{k-1}|S_k
\]

Cette formule calcule le résultat financier de la couverture en prenant en compte les gains ou pertes sur les actions et les coûts de transaction.

Avec :

- \(Z\) : flux financier lié à l'option et à sa prime, selon la convention retenue.
- \(\delta_k\) : quantité d'actions détenue à l'instant \(k\).
- \(S_k\) : prix de l'action.
- \(c\) : taux de coût de transaction.

**D. Objectif du Deep Hedging**

\[
\min\_{\theta}\operatorname{Var}(P\&L)
\]

Le réseau de neurones cherche les paramètres optimaux \(\theta\) permettant de minimiser la variance du P&L et donc de réduire la dispersion des résultats financiers.

La stratégie est définie par :

\[
\delta*k=f*\theta\left(
\ln(S*k/K),T-t_k,\delta*{k-1}
\right)
\]

Le réseau utilise le prix relatif, le temps restant et la position précédente pour déterminer la nouvelle quantité d'actions à détenir.

## 3. Lien avec le TP 0 : intégration dans Odoo

Le TP 0 constitue la base de notre application métier. Le module existant `cours_derives` sera étendu afin d'intégrer la simulation des stratégies de couverture.

Un nouveau modèle Odoo, nommé `cours.simulation.couverture`, pourra être ajouté pour gérer les simulations. Il contiendra notamment les paramètres suivants : prix initial, prix d'exercice, maturité, coûts de transaction et nombre de trajectoires.

Le fonctionnement prévu est le suivant :

1. L'utilisateur saisit les paramètres financiers dans une interface Odoo.
2. Il lance la simulation à l'aide d'un bouton.
3. Le moteur Python calcule les résultats des trois stratégies et utilise le modèle Deep Hedging préalablement entraîné avec PyTorch.
4. Les résultats sont enregistrés dans PostgreSQL à travers Odoo.
5. Un tableau de bord affiche les indicateurs financiers : moyenne du P&L, écart-type, VaR et graphiques comparatifs.

Cette intégration permet de transformer le module `cours_derives` en un outil de simulation et de comparaison des stratégies de couverture.

## 4. Algorithme de calcul en Python

Le traitement est organisé en cinq étapes :

1. **Simulation des prix :** générer plusieurs trajectoires du sous-jacent à l'aide du modèle GBM (_Geometric Brownian Motion_), avec NumPy.
2. **Calcul du P&L :** calculer le payoff de l'option, les gains ou pertes et les coûts de transaction pour chaque stratégie.
3. **Entraînement du réseau :** construire un réseau de neurones avec PyTorch et ajuster ses paramètres pour minimiser la variance du P&L.
4. **Calcul des stratégies :** appliquer la couverture sans intervention, la couverture Delta classique et la stratégie Deep Hedging.
5. **Comparaison :** évaluer les performances des trois méthodes sur les mêmes trajectoires de test et analyser leurs résultats.

## 5. Conclusion

Le projet Deep Hedging vise à étudier l'apport de l'intelligence artificielle dans la gestion des risques financiers. L'objectif est de déterminer si une stratégie apprise par un réseau de neurones peut améliorer la couverture classique de Black-Scholes, notamment lorsque les coûts de transaction sont pris en compte.

**Technologies :** Python, NumPy, Pandas, PyTorch, Odoo 17, PostgreSQL et Docker.
