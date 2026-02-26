from typing import Tuple, Dict
import numpy as np

try:
    from .common import get_default_params, AnalyticalPricer
except ImportError:
    from common import get_default_params, AnalyticalPricer


def test_put_call_parity() -> Tuple[bool, str, Dict]:
    params = get_default_params()
    K = params.S0
    T = 1.0

    pricer = AnalyticalPricer(params)

    call_price = pricer.call_price(K, T)
    put_price = pricer.put_price(K, T)

    parity_theoretical = params.S0 * np.exp(-params.q * T) - K * np.exp(-params.r * T)
    parity_actual = call_price - put_price
    error = abs(parity_actual - parity_theoretical)

    passed = error < 0.01
    message = f"Error = {error:.6f}"
    details = {
        'call': call_price,
        'put': put_price,
        'theoretical': parity_theoretical,
        'actual': parity_actual,
        'error': error
    }

    return passed, message, details
