"""
HedgeNet : réseau de neurones qui apprend une stratégie de couverture (Deep Hedging).

Ce fichier réutilise le code d'Aymane :
- src/data.py    -> simulate_gbm() donne les trajectoires S de forme (n_paths, n_steps + 1)
- src/pricing.py -> compute_hedging_pnl() calcule le P&L avec les coûts de transaction

Le réseau renvoie des deltas de forme (n_paths, n_steps), comme bs_delta_strategy(),
donc les 3 stratégies (naïve, Delta BS, HedgeNet) se comparent avec les mêmes fonctions.
"""

import torch
import torch.nn as nn

try:
    from src.pricing import compute_hedging_pnl  # import depuis la racine du dépôt (notebooks)
except ImportError:
    from pricing import compute_hedging_pnl  # import depuis le dossier src/


class HedgeNet(nn.Module):
    """
    Petit réseau (MLP) à 2 couches cachées.

    Entrées (3) : ln(S_k / K), temps restant T - t_k, delta précédent delta_{k-1}
    Sortie  (1) : nouveau delta_k, entre 0 et 1 grâce à la Sigmoid
                  (pour un Call vendu, la couverture est entre 0 et 1 action).
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (n_paths, 3)  ->  sortie : (n_paths,)
        return self.net(x).squeeze(-1)


def hedgenet_strategy(model: HedgeNet, S: torch.Tensor, K: float, T: float) -> torch.Tensor:
    """
    Stratégie 3 : couverture apprise par le réseau.

    Calcule delta_k = f_theta(ln(S_k/K), T - t_k, delta_{k-1}) pour k = 0 ... n-1.
    Le calcul se fait pas à pas, car chaque delta dépend du delta précédent.
    Retourne un tenseur de forme (n_paths, n_steps).
    """
    n_paths, n_plus_1 = S.shape
    n = n_plus_1 - 1
    dt = T / n

    delta_prev = torch.zeros(n_paths, dtype=S.dtype, device=S.device)  # delta_{-1} = 0
    deltas = []

    for k in range(n):
        tau = T - k * dt  # temps restant : T, T - dt, ..., dt (comme dans bs_delta_strategy)
        x = torch.stack(
            [
                torch.log(S[:, k] / K),
                torch.full_like(delta_prev, tau),
                delta_prev,
            ],
            dim=1,
        )
        delta_k = model(x)
        deltas.append(delta_k)
        delta_prev = delta_k

    return torch.stack(deltas, dim=1)


def train_hedgenet(
    S_train: torch.Tensor,
    K: float,
    T: float,
    c: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 5_000,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[HedgeNet, list[float]]:
    """
    Entraînement simple : on cherche les poids qui minimisent la variance du P&L.

    Version S2 (test rapide) : mêmes trajectoires à chaque epoch, pas d'early stopping.
    En S3 : nouvelles trajectoires à chaque epoch + early stopping sur un jeu de validation.
    """
    torch.manual_seed(seed)
    model = HedgeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        order = torch.randperm(S_train.shape[0])  # on mélange les trajectoires
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, S_train.shape[0], batch_size):
            S_batch = S_train[order[start : start + batch_size]]

            deltas = hedgenet_strategy(model, S_batch, K, T)
            pnl, _ = compute_hedging_pnl(S_batch, deltas, K, c)
            loss = pnl.var()  # objectif : min Var(P&L)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / n_batches)
        print(f"Epoch {epoch + 1}/{epochs} - Var(P&L) = {losses[-1]:.4f}")

    return model, losses
