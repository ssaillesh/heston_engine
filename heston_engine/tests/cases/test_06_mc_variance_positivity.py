from typing import Tuple, Dict
import numpy as np

try:
    from .common import HestonParams, MonteCarloSimulator
except ImportError:
    from common import HestonParams, MonteCarloSimulator


def test_mc_variance_positivity() -> Tuple[bool, str, Dict]:
    params = HestonParams(
        kappa=1.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
        r=0.05,
        q=0.02,
        S0=100.0,
        V0=0.04
    )

    mc = MonteCarloSimulator(params)
    _, V_paths = mc.simulate_paths_qe(T=1.0, N_steps=252, N_paths=1000)

    min_variance = np.min(V_paths)
    all_positive = min_variance >= 0

    passed = all_positive
    message = f"Min variance = {min_variance:.6f}"
    details = {
        'min_variance': min_variance,
        'all_positive': all_positive,
        'feller_ratio': params.feller_ratio
    }

    return passed, message, details
