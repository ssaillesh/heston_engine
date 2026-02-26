from typing import Tuple, Dict

try:
    from .common import get_default_params, AnalyticalPricer, implied_volatility_newton
except ImportError:
    from common import get_default_params, AnalyticalPricer, implied_volatility_newton


def test_moneyness_effects() -> Tuple[bool, str, Dict]:
    params = get_default_params()
    T = 1.0

    pricer = AnalyticalPricer(params)

    strikes = [80, 90, 100, 110, 120]
    prices = []
    impl_vols = []

    for K in strikes:
        price = pricer.call_price(K, T)
        iv = implied_volatility_newton(price, params.S0, K, T, params.r, params.q, 'call')
        prices.append(price)
        impl_vols.append(iv)

    prices_decreasing = all(prices[i] > prices[i+1] for i in range(len(prices)-1))
    has_skew = impl_vols[0] > impl_vols[-1]

    passed = prices_decreasing and has_skew
    message = f"Price monotonic: {prices_decreasing}, Skew present: {has_skew}"
    details = {
        'strikes': strikes,
        'prices': prices,
        'implied_vols': impl_vols,
        'prices_decreasing': prices_decreasing,
        'has_skew': has_skew
    }

    return passed, message, details
