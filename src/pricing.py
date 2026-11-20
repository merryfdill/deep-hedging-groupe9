import math
import torch


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Fonction de répartition N(x) de la loi normale centrée réduite."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Prix analytique d'un Call européen selon Black-Scholes."""
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    n_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    n_d2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    return S0 * n_d1 - K * math.exp(-r * T) * n_d2


def bs_delta_strategy(
    S: torch.Tensor, K: float, T: float, r: float, sigma: float
) -> torch.Tensor:
    """
    Stratégie 1 : Couverture Delta classique de Black-Scholes.
    Calcule delta_k = N(d1(S_k, T - t_k)) pour k = 0 ... n-1.
    """
    _, n_plus_1 = S.shape
    n = n_plus_1 - 1
    dt = T / n

    # Temps restant jusqu'à l'échéance à chaque pas : [T, T-dt, ..., dt]
    tau = torch.linspace(T, dt, steps=n, dtype=S.dtype, device=S.device).unsqueeze(0)
    S_t = S[:, :-1]

    d1 = (torch.log(S_t / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    return norm_cdf(d1)


def naive_strategy(S: torch.Tensor, mode: str = "no_hedge", seed: int = 42) -> torch.Tensor:
    """
    Stratégie 2 : Stratégie naïve de référence.
    - 'no_hedge' : aucune couverture (delta_k = 0)
    - 'random'   : couverture aléatoire uniforme dans [0, 1]
    """
    M, n_plus_1 = S.shape
    n = n_plus_1 - 1
    if mode == "no_hedge":
        return torch.zeros((M, n), dtype=S.dtype, device=S.device)
    if mode == "random":
        torch.manual_seed(seed)
        return torch.rand((M, n), dtype=S.dtype, device=S.device)
    raise ValueError("Mode invalide : choisir 'no_hedge' ou 'random'.")


def compute_hedging_pnl(
    S: torch.Tensor,
    deltas: torch.Tensor,
    K: float,
    c: float = 1e-3,
    include_premium: bool = False,
    premium: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calcule le P&L final et le coût de transaction total sur chaque trajectoire.
    Formule du guide :
      P/L = -Z + sum_{k=0}^{n-1} delta_k*(S_{k+1} - S_k) - sum_{k=0}^{n-1} c*|delta_k - delta_{k-1}|*S_k
    """
    # Gains / pertes cumulés sur le sous-jacent
    price_diffs = S[:, 1:] - S[:, :-1]
    trading_gains = torch.sum(deltas * price_diffs, dim=1)

    # Coûts de transaction proportionnels (avec delta_{-1} = 0)
    delta_prev = torch.cat(
        [torch.zeros((deltas.shape[0], 1), dtype=deltas.dtype, device=deltas.device), deltas[:, :-1]],
        dim=1,
    )
    transaction_costs = torch.sum(c * torch.abs(deltas - delta_prev) * S[:, :-1], dim=1)

    # Payoff de l'option Call vendue à maturité : Z = max(S_n - K, 0)
    payoff = torch.relu(S[:, -1] - K)

    pnl = -payoff + trading_gains - transaction_costs
    if include_premium:
        pnl = pnl + premium

    return pnl, transaction_costs


def value_at_risk(pnl: torch.Tensor, alpha: float = 0.95) -> float:
    """Retourne la VaR : perte positive dépassée avec une probabilité 1-alpha."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha doit être strictement compris entre 0 et 1.")
    quantile = torch.quantile(pnl, 1.0 - alpha)
    return (-quantile).item()


def conditional_value_at_risk(pnl: torch.Tensor, alpha: float = 0.95) -> float:
    """Retourne la perte moyenne des scénarios dont la perte dépasse la VaR."""
    var = value_at_risk(pnl, alpha)
    losses = -pnl.detach()
    tail_losses = losses[losses >= var]
    if tail_losses.numel() == 0:
        raise ValueError("Impossible de calculer le CVaR : aucun scénario dans la queue.")
    return tail_losses.mean().item()


def entropic_risk(pnl: torch.Tensor, lambda_risk: float) -> float:
    """Retourne rho_lambda(P&L) = log(E(exp(-lambda P&L))) / lambda."""
    if lambda_risk <= 0.0:
        raise ValueError("lambda_risk doit être strictement positif.")
    exponent = -lambda_risk * pnl.detach()
    log_mgf = torch.logsumexp(exponent, dim=0) - math.log(pnl.numel())
    return (log_mgf / lambda_risk).item()


def summarize_pnl(
    pnl: torch.Tensor,
    costs: torch.Tensor,
    alpha: float = 0.95,
    lambda_risk: float | None = None,
) -> dict[str, float]:
    """Calcule les indicateurs clés demandés dans le guide.

    La VaR est rapportée comme une perte positive. Elle correspond donc
    à l'opposé du quantile à 5 % du P&L.
    """
    quantile = torch.quantile(pnl, 1.0 - alpha).item()
    metrics = {
        "Moyenne P&L": pnl.mean().item(),
        "Écart-type P&L": pnl.std().item(),
        f"Quantile {int((1.0 - alpha) * 100)}%": quantile,
        f"VaR {int(alpha * 100)}%": value_at_risk(pnl, alpha),
        f"CVaR {int(alpha * 100)}%": conditional_value_at_risk(pnl, alpha),
        "Coût transaction moyen": costs.mean().item(),
    }
    if lambda_risk is not None:
        metrics["Risque entropique"] = entropic_risk(pnl, lambda_risk)
    return metrics