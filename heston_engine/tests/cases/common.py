import os
import sys
import numpy as np
from typing import Tuple, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.parameters import HestonParams, get_default_params
from backend.core.grid import Grid, create_default_grid
from backend.core.boundaries import BoundaryConditions, implied_volatility_newton
from backend.solvers.pde_solver import PDESolver
from backend.solvers.analytical import AnalyticalPricer
from backend.solvers.monte_carlo import MonteCarloSimulator
from backend.greeks.calculator import GreeksCalculator

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except Exception:
    QUANTLIB_AVAILABLE = False


def is_numerically_stable(value: float, bound: float = 1e6) -> bool:
    return np.isfinite(value) and abs(value) <= bound


def quantlib_heston_call_price(params: HestonParams, K: float, T: float) -> float:
    if not QUANTLIB_AVAILABLE:
        raise RuntimeError("QuantLib is not installed")

    import QuantLib as ql

    evaluation_date = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = evaluation_date
    day_count = ql.Actual365Fixed()

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(params.S0))
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, params.r, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, params.q, day_count)
    )

    heston_process = ql.HestonProcess(
        risk_free_ts,
        dividend_ts,
        spot_handle,
        params.V0,
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
    )
    model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(model)

    maturity_date = evaluation_date + int(round(T * 365))
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)

    return float(option.NPV())


def compute_benchmark_score(analytical_mape_pct: float, analytical_max_rel_pct: float, mc_rel_pct: float) -> float:
    penalty = (
        40.0 * analytical_mape_pct +
        15.0 * analytical_max_rel_pct +
        5.0 * mc_rel_pct
    )
    return max(0.0, 100.0 - penalty)
