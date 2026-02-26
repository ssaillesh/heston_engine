from typing import Tuple, Dict
import numpy as np

try:
    from .common import HestonParams, AnalyticalPricer, BoundaryConditions
except ImportError:
    from common import HestonParams, AnalyticalPricer, BoundaryConditions


def test_bs_limit() -> Tuple[bool, str, Dict]:
    params = HestonParams(
        kappa=5.0,
        theta=0.04,
        sigma=0.001,
        rho=0.0,
        r=0.05,
        q=0.02,
        S0=100.0,
        V0=0.04
    )

    K = 100.0
    T = 1.0

    pricer = AnalyticalPricer(params)
    heston_price = pricer.call_price(K, T)

    bc = BoundaryConditions()
    bs_sigma = np.sqrt(params.theta)
    bs_price = bc.black_scholes_call(params.S0, K, T, params.r, params.q, bs_sigma)

    error = abs(heston_price - bs_price)
    rel_error = error / bs_price * 100

    passed = rel_error < 1.0
    message = f"Heston={heston_price:.4f}, BS={bs_price:.4f}, Diff={rel_error:.2f}%"
    details = {
        'heston': heston_price,
        'bs': bs_price,
        'error': error,
        'rel_error_pct': rel_error
    }

    return passed, message, details
