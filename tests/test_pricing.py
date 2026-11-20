import torch
import pytest
from src.pricing import (
    bs_call_price,
    bs_delta_strategy,
    compute_hedging_pnl,
    summarize_pnl,
    value_at_risk,
    conditional_value_at_risk,
    entropic_risk,
)

def test_bs_call_price():
    # Example: S=100, K=100, T=1, r=0.03, sigma=0.2
    price = bs_call_price(100.0, 100.0, 1.0, 0.03, 0.2)
    # Price should be positive and roughly around theoretical value (~10)
    assert price > 0
    assert 9.4 < price < 9.5

def test_bs_delta_strategy_range():
    # Delta should be between 0 and 1 for a call
    S = torch.tensor([[100.0, 105.0, 110.0]], dtype=torch.float32)
    K = 100.0
    T = 1.0
    r = 0.03
    sigma = 0.2

    deltas = bs_delta_strategy(S, K, T, r, sigma)
    assert deltas.shape == (1, 2) # n_steps=2
    assert (deltas >= 0).all()
    assert (deltas <= 1).all()

def test_pnl_calculation():
    # Simple case
    S = torch.tensor([[100.0, 101.0, 102.0]], dtype=torch.float32)
    deltas = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    K = 100.0
    c = 0.0

    pnl, costs = compute_hedging_pnl(S, deltas, K, c=c)

    # Expected:
    # Trading gains: 0.5 * (101-100) + 0.5 * (102-101) = 0.5 + 0.5 = 1.0
    # Payoff: max(102-100, 0) = 2.0
    # P&L = -2.0 + 1.0 = -1.0
    assert torch.isclose(pnl[0], torch.tensor(-1.0))
    assert torch.isclose(costs[0], torch.tensor(0.0))


def test_transaction_costs():
    S = torch.tensor([[100.0, 110.0]], dtype=torch.float32)
    deltas = torch.tensor([[0.5]], dtype=torch.float32)

    pnl, costs = compute_hedging_pnl(S, deltas, K=100.0, c=0.01)

    assert torch.isclose(pnl[0], torch.tensor(-5.5))
    assert torch.isclose(costs[0], torch.tensor(0.5))


def test_summarize_pnl_var_and_cvar():
    pnl = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
    costs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    metrics = summarize_pnl(pnl, costs)

    assert torch.isclose(
        torch.tensor(metrics["Quantile 5%"]),
        torch.tensor(-9.0),
    )
    assert torch.isclose(torch.tensor(metrics["VaR 95%"]), torch.tensor(9.0))
    assert torch.isclose(torch.tensor(metrics["CVaR 95%"]), torch.tensor(10.0))


def test_risk_functions():
    pnl = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])

    assert value_at_risk(pnl) == pytest.approx(9.0)
    assert conditional_value_at_risk(pnl) == pytest.approx(10.0)
    assert entropic_risk(torch.zeros(5), 1.0) == pytest.approx(0.0)
