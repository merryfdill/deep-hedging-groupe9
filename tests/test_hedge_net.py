import torch
import pytest
from src.hedge_net import HedgeNet, hedgenet_strategy

def test_hedgenet_forward():
    model = HedgeNet(hidden=16)
    x = torch.randn(5, 3) # (n_paths, 3)
    output = model(x)
    assert output.shape == (5,)
    assert (output >= 0).all()
    assert (output <= 1).all()

def test_hedgenet_strategy_shape():
    model = HedgeNet(hidden=16)
    S = torch.randn(10, 31) # (n_paths, n_steps + 1)
    K = 100.0
    T = 1.0

    deltas = hedgenet_strategy(model, S, K, T)
    assert deltas.shape == (10, 30) # (n_paths, n_steps)
