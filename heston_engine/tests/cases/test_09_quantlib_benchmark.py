from typing import Tuple, Dict, List
import numpy as np

try:
    from .common import (
        QUANTLIB_AVAILABLE,
        get_default_params,
        AnalyticalPricer,
        MonteCarloSimulator,
        quantlib_heston_call_price,
        compute_benchmark_score,
    )
except ImportError:
    from common import (
        QUANTLIB_AVAILABLE,
        get_default_params,
        AnalyticalPricer,
        MonteCarloSimulator,
        quantlib_heston_call_price,
        compute_benchmark_score,
    )


def test_quantlib_benchmark_alignment() -> Tuple[bool, str, Dict]:
    if not QUANTLIB_AVAILABLE:
        return True, "QuantLib not installed (benchmark skipped)", {'skipped': True}

    params = get_default_params()
    pricer = AnalyticalPricer(params)

    scenarios: List[Tuple[float, float]] = [
        (80.0, 0.5),
        (90.0, 1.0),
        (100.0, 1.0),
        (110.0, 1.5),
        (120.0, 2.0),
    ]

    rows: List[Dict[str, float]] = []
    rel_errors_pct: List[float] = []
    abs_errors: List[float] = []

    for K, T in scenarios:
        model_price = pricer.call_price(K, T)
        ql_price = quantlib_heston_call_price(params, K, T)
        abs_err = abs(model_price - ql_price)
        rel_pct = abs_err / max(1e-8, ql_price) * 100

        rows.append({
            'K': K,
            'T': T,
            'model': model_price,
            'quantlib': ql_price,
            'abs_error': abs_err,
            'rel_error_pct': rel_pct,
        })
        abs_errors.append(abs_err)
        rel_errors_pct.append(rel_pct)

    analytical_rmse = float(np.sqrt(np.mean(np.square(abs_errors))))
    analytical_mape_pct = float(np.mean(rel_errors_pct))
    analytical_max_rel_pct = float(np.max(rel_errors_pct))

    mc = MonteCarloSimulator(params)
    mc_price, mc_stderr = mc.price_european_call(100.0, 1.0, 252, 100000, seed=42)
    ql_atm = quantlib_heston_call_price(params, 100.0, 1.0)
    mc_rel_pct = abs(mc_price - ql_atm) / max(1e-8, ql_atm) * 100

    score = compute_benchmark_score(analytical_mape_pct, analytical_max_rel_pct, mc_rel_pct)

    passed = (
        analytical_mape_pct < 0.10 and
        analytical_max_rel_pct < 0.50 and
        mc_rel_pct < 1.50
    )

    if score >= 99.5:
        quality = "perfect"
    elif score >= 95.0:
        quality = "excellent"
    elif score >= 85.0:
        quality = "low-error"
    else:
        quality = "needs-improvement"

    message = (
        f"Score={score:.2f}/100 ({quality}), "
        f"MAPE={analytical_mape_pct:.4f}%, "
        f"MaxRel={analytical_max_rel_pct:.4f}%, MC={mc_rel_pct:.4f}%"
    )

    details = {
        'analytical_rmse': analytical_rmse,
        'analytical_mape_pct': analytical_mape_pct,
        'analytical_max_rel_pct': analytical_max_rel_pct,
        'mc_rel_error_pct': mc_rel_pct,
        'mc_stderr': mc_stderr,
        'score': score,
        'quality': quality,
        'rows': rows,
    }

    return passed, message, details
