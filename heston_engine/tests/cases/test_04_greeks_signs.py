from typing import Tuple, Dict

try:
    from .common import get_default_params, GreeksCalculator
except ImportError:
    from common import get_default_params, GreeksCalculator


def test_greeks_signs() -> Tuple[bool, str, Dict]:
    params = get_default_params()
    K = params.S0
    T = 1.0

    calculator = GreeksCalculator(params)
    greeks = calculator.all_greeks(K, T, 'call')

    checks = {
        'delta > 0': greeks['delta'] > 0,
        'gamma > 0': greeks['gamma'] > 0,
        'vega > 0': greeks['vega'] > 0,
        'theta < 0': greeks['theta'] < 0,
        'rho > 0': greeks['rho'] > 0
    }

    passed = all(checks.values())
    message = "All signs correct" if passed else f"Failed: {[k for k, v in checks.items() if not v]}"
    details = {**greeks, 'checks': checks}

    return passed, message, details
