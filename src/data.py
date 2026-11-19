import math
import torch


def simulate_gbm(
    S0: float = 100.0,
    r: float = 0.03,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 30,
    n_paths: int = 100_000,
    seed: int | None = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Simule des trajectoires de prix sous probabilité risque-neutre selon un
    Mouvement Brownien Géométrique (GBM) vectorisé en PyTorch.

    Retourne un tenseur float32 de forme (n_paths, n_steps + 1).
    """
    if seed is not None:
        torch.manual_seed(seed)

    dt = T / n_steps
    Z = torch.randn((n_paths, n_steps), dtype=torch.float32, device=device)

    # Schéma exact du GBM : S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
    log_prices = math.log(S0) + torch.cumsum(log_returns, dim=1)
    prices = torch.exp(log_prices)

    # Ajout du prix initial S0 à t = 0
    S0_col = torch.full((n_paths, 1), float(S0), dtype=torch.float32, device=device)
    return torch.cat([S0_col, prices], dim=1)


def get_train_test_datasets(
    S0: float = 100.0,
    r: float = 0.03,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 30,
    train_paths: int = 100_000,
    test_paths: int = 20_000,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Génère les jeux d'entraînement (100k) et de test (20k) reproductibles."""
    train_data = simulate_gbm(S0, r, sigma, T, n_steps, train_paths, seed=42, device=device)
    test_data = simulate_gbm(S0, r, sigma, T, n_steps, test_paths, seed=123, device=device)
    return train_data, test_data


if __name__ == "__main__":
    train_S, test_S = get_train_test_datasets(n_steps=30)
    print(f"Train shape : {train_S.shape} | dtype : {train_S.dtype}")
    print(f"Test shape  : {test_S.shape}  | dtype : {test_S.dtype}")