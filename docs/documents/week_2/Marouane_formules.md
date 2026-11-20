# Formules – Projet 9 : Deep Hedging (T07)

Notations : $S_k$ prix du sous-jacent à la date $t_k = k\,\Delta t$, $\Delta t = T/n$, $K$ strike, $r$ taux sans risque, $\sigma$ volatilité, $c$ coût de transaction proportionnel, $\delta_k$ nombre d'actions détenues entre $t_k$ et $t_{k+1}$ (avec $\delta_{-1}=0$).

## 1. Marché simulé (GBM sous probabilité risque-neutre)

$$S_{k+1} = S_k \exp\!\Big[\big(r-\tfrac{\sigma^2}{2}\big)\Delta t + \sigma\sqrt{\Delta t}\,Z_k\Big],\qquad Z_k \sim \mathcal N(0,1)$$

Schéma exact du mouvement brownien géométrique : on génère beaucoup de scénarios de prix pour tester les stratégies. (Code : `simulate_gbm` dans `src/data.py`.)

## 2. Option vendue et Delta de Black-Scholes

$$Z = \max(S_n - K,\,0) \qquad \text{(payoff du Call vendu, à payer à l'échéance)}$$

$$\Delta_k = N(d_1),\qquad d_1=\frac{\ln(S_k/K)+\big(r+\tfrac{\sigma^2}{2}\big)(T-t_k)}{\sigma\sqrt{T-t_k}}$$

Le Delta est le nombre d'actions à détenir à la date $t_k$ pour neutraliser la sensibilité de l'option au prix. Il se recalcule à chaque date avec le prix courant $S_k$ et le temps restant $T-t_k$. (Code : `bs_delta_strategy`.)

## 3. P&L de la couverture avec coûts de transaction

$$\text{P\&L} \;=\; -Z \;+\; \sum_{k=0}^{n-1}\delta_k\,(S_{k+1}-S_k)\;-\;\sum_{k=0}^{n-1} c\,\big|\delta_k-\delta_{k-1}\big|\,S_k$$

- $-Z$ : on paie le payoff de l'option vendue.
- $\sum \delta_k (S_{k+1}-S_k)$ : gains ou pertes sur les actions détenues.
- $\sum c\,|\delta_k-\delta_{k-1}|\,S_k$ : coût payé à chaque réajustement (proportionnel au montant échangé).

**Convention de prime :** la formule du guide ne contient pas la prime. Si on veut inclure la prime encaissée à la vente, on ajoute $+\,\text{prime}$ (c'est l'option `include_premium=True` dans `compute_hedging_pnl`). On précise la convention retenue dans chaque résultat. (Code : `compute_hedging_pnl` dans `src/pricing.py`.)

## 4. Mesures de risque

Soit $L=-\text{P\&L}$ la perte.

**Variance** (critère d'entraînement principal) :

$$\min_\theta \ \mathrm{Var}(\text{P\&L})$$

Une variance faible signifie un résultat peu dispersé : une bonne couverture.

**VaR à 95 %** : perte dépassée dans seulement 5 % des cas.

$$\mathrm{VaR}_{95\%} = -\,q_{5\%}(\text{P\&L})$$

où $q_{5\%}$ est le quantile à 5 % du P&L. (Dans `summarize_pnl`, la ligne « Quantile 5 % » donne $q_{5\%}$ ; la VaR est son opposé.)

**CVaR** (perte moyenne dans les pires cas) :

$$\mathrm{CVaR}_\alpha = \mathbb E\big[L \mid L \ge \mathrm{VaR}_\alpha\big] = \min_{w}\Big\{\,w+\tfrac{1}{1-\alpha}\,\mathbb E\big[(L-w)^+\big]\Big\}$$

La seconde écriture (Rockafellar-Uryasev) est celle qui se minimise facilement dans un réseau de neurones.

**Risque entropique** (pénalise fortement les grosses pertes) :

$$\rho_\lambda(\text{P\&L}) = \frac{1}{\lambda}\ln \mathbb E\big[e^{-\lambda\,\text{P\&L}}\big],\qquad \lambda>0$$

## 5. Stratégie apprise (réseau de neurones)

$$\delta_k = f_\theta\big(\ln(S_k/K),\; T-t_k,\; \delta_{k-1}\big)$$

Le réseau reçoit le prix relatif, le temps restant et la position précédente, et renvoie la nouvelle position. Le $\delta_{k-1}$ en entrée lui permet de tenir compte des coûts : réajuster moins quand $c$ augmente. (Code : `HedgeNet` dans `src/hedge_net.py`, sortie bornée dans $[0,1]$ par une Sigmoid.)

## 6. Indicateurs de comparaison (3 stratégies, mêmes trajectoires de test)

| Indicateur | Formule |
|---|---|
| Moyenne du P&L | $\overline{\text{P\&L}}$ |
| Écart-type du P&L | $\sqrt{\mathrm{Var}(\text{P\&L})}$ |
| Quantile 5 % | $q_{5\%}(\text{P\&L})$ |
| Coût de transaction total | $\sum_k c\,\lvert\delta_k-\delta_{k-1}\rvert S_k$ |

Stratégies comparées : sans couverture ($\delta_k=0$), Delta Black-Scholes, HedgeNet.

## 7. Hypothèses

- Marché simulé (GBM) : le réseau n'est valable que dans cette dynamique d'entraînement.
- $r$, $\sigma$ constants ; coût $c$ proportionnel et connu ; fréquence de rééquilibrage fixe.
- Trajectoires d'entraînement indépendantes ; jeux d'entraînement et de test séparés pour mesurer la généralisation.
