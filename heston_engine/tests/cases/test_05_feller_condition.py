from typing import Tuple, Dict

try:
    from .common import HestonParams
except ImportError:
    from common import HestonParams


def test_feller_condition() -> Tuple[bool, str, Dict]:
    params_ok = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, r=0.05, q=0.02, S0=100.0, V0=0.04)
    params_bad = HestonParams(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, r=0.05, q=0.02, S0=100.0, V0=0.04)

    checks = {
        'ok_satisfied': params_ok.feller_satisfied,
        'ok_ratio > 1': params_ok.feller_ratio > 1,
        'bad_violated': not params_bad.feller_satisfied,
        'bad_ratio < 1': params_bad.feller_ratio < 1
    }

    passed = all(checks.values())
    message = f"OK ratio={params_ok.feller_ratio:.2f}, Bad ratio={params_bad.feller_ratio:.2f}"
    details = {
        'params_ok_ratio': params_ok.feller_ratio,
        'params_bad_ratio': params_bad.feller_ratio,
        'checks': checks
    }

    return passed, message, details
