import math
import torch
import pytest
from src.data import simulate_gbm, get_train_test_datasets


def test_simulate_gbm_shape_and_dtype():
    n_paths = 500
    n_steps = 20
    S = simulate_gbm(n_paths=n_paths, n_steps=n_steps, seed=42)

    assert isinstance(S, torch.Tensor)
    assert S.dtype == torch.float32
    assert S.shape == (n_paths, n_steps + 1)


def test_simulate_gbm_initial_prices():
    S0 = 120.0
    n_paths = 100
    S = simulate_gbm(S0=S0, n_paths=n_paths, seed=42)

    # Initial price across all paths must equal S0
    assert torch.allclose(S[:, 0], torch.tensor(S0, dtype=torch.float32))


def test_simulate_gbm_prices_positive():
    S = simulate_gbm(n_paths=1000, n_steps=50, seed=42)
    assert (S > 0).all()


def test_simulate_gbm_reproducibility():
    S1 = simulate_gbm(n_paths=100, n_steps=10, seed=42)
    S2 = simulate_gbm(n_paths=100, n_steps=10, seed=42)
    S3 = simulate_gbm(n_paths=100, n_steps=10, seed=99)

    assert torch.equal(S1, S2)
    assert not torch.equal(S1, S3)


def test_simulate_gbm_statistical_mean():
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_paths = 100_000

    S = simulate_gbm(S0=S0, r=r, sigma=sigma, T=T, n_steps=50, n_paths=n_paths, seed=42)

    # Theoretical expected price E[S_T] = S0 * exp(r * T)
    expected_S_T = S0 * math.exp(r * T)
    empirical_mean = S[:, -1].mean().item()

    # Relative tolerance of ~1% for 100k paths
    assert math.isclose(empirical_mean, expected_S_T, rel_tol=0.015)


def test_get_train_test_datasets():
    train_paths = 1000
    test_paths = 200
    n_steps = 15

    train_data, test_data = get_train_test_datasets(
        train_paths=train_paths, test_paths=test_paths, n_steps=n_steps
    )

    assert train_data.shape == (train_paths, n_steps + 1)
    assert test_data.shape == (test_paths, n_steps + 1)
    # Train and test datasets should be generated with different seeds
    assert not torch.equal(train_data[:test_paths], test_data)
