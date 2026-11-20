import sys
from pathlib import Path
import pytest
import torch

# Ensure src module is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import simulate_gbm
from src.hedge_net import HedgeNet


@pytest.fixture
def default_market_params():
    return {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.03,
        "sigma": 0.2,
        "n_steps": 30,
    }


@pytest.fixture
def sample_gbm_paths(default_market_params):
    return simulate_gbm(
        S0=default_market_params["S0"],
        r=default_market_params["r"],
        sigma=default_market_params["sigma"],
        T=default_market_params["T"],
        n_steps=default_market_params["n_steps"],
        n_paths=1000,
        seed=42,
    )


@pytest.fixture
def hedge_net_model():
    return HedgeNet(hidden=32)
