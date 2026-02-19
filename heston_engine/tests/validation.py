"""
═══════════════════════════════════════════════════════════════════════════════
HESTON ENGINE VALIDATION SUITE
═══════════════════════════════════════════════════════════════════════════════

Automated tests to validate the Heston model implementation:
1. Put-Call Parity: C - P = S·e^{-qT} - K·e^{-rT}
2. Black-Scholes Limit: When σ → 0, Heston → BS
3. Method Convergence: All three methods should give similar prices
4. Parameter Sensitivity: Greeks have correct signs
5. Monte Carlo Statistics: Mean and variance converge

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import numpy as np
from typing import Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.parameters import HestonParams, get_default_params
from backend.core.grid import Grid, create_default_grid
from backend.core.boundaries import BoundaryConditions, implied_volatility_newton
from backend.solvers.pde_solver import PDESolver
from backend.solvers.analytical import AnalyticalPricer
from backend.solvers.monte_carlo import MonteCarloSimulator
from backend.greeks.calculator import GreeksCalculator


# ═══════════════════════════════════════════════════════════════════════════════
# TEST UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str, passed: bool, message: str, details: Optional[dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.name} - {self.message}"


def run_test(name: str, test_func) -> TestResult:
    """Execute a test function and return result."""
    try:
        passed, message, details = test_func()
        return TestResult(name, passed, message, details)
    except Exception as e:
        return TestResult(name, False, f"Exception: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: PUT-CALL PARITY
# ═══════════════════════════════════════════════════════════════════════════════

def test_put_call_parity() -> Tuple[bool, str, dict]:
    """
    Test Put-Call Parity:
    
    C - P = S·e^{-qT} - K·e^{-rT}
    
    This fundamental arbitrage relationship must hold for any valid
    option pricing model.
    """
    params = get_default_params()
    K = params.S0
    T = 1.0
    
    pricer = AnalyticalPricer(params)
    
    call_price = pricer.call_price(K, T)
    put_price = pricer.put_price(K, T)
    
    # Theoretical parity value
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


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: BLACK-SCHOLES LIMIT
# ═══════════════════════════════════════════════════════════════════════════════

def test_bs_limit() -> Tuple[bool, str, dict]:
    """
    Test Black-Scholes Limit:
    
    When volatility of volatility σ → 0 and V₀ = θ (constant vol),
    the Heston model should converge to Black-Scholes.
    
    In this limit, the variance process becomes deterministic:
    V(t) → θ for all t
    """
    # Create params with very low vol-of-vol
    params = HestonParams(
        kappa=5.0,      # Fast mean reversion
        theta=0.04,     # Long-term variance = 0.04 → volatility = 0.2
        sigma=0.001,    # Very low vol-of-vol (near BS limit)
        rho=0.0,        # No correlation
        r=0.05,
        q=0.02,
        S0=100.0,
        V0=0.04         # Start at long-term level
    )
    
    K = 100.0
    T = 1.0
    
    # Heston price
    pricer = AnalyticalPricer(params)
    heston_price = pricer.call_price(K, T)
    
    # Black-Scholes price with σ = √θ
    bc = BoundaryConditions()
    bs_sigma = np.sqrt(params.theta)
    bs_price = bc.black_scholes_call(params.S0, K, T, params.r, params.q, bs_sigma)
    
    error = abs(heston_price - bs_price)
    rel_error = error / bs_price * 100
    
    passed = rel_error < 1.0  # Within 1%
    message = f"Heston={heston_price:.4f}, BS={bs_price:.4f}, Diff={rel_error:.2f}%"
    details = {
        'heston': heston_price,
        'bs': bs_price,
        'error': error,
        'rel_error_pct': rel_error
    }
    
    return passed, message, details


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: METHOD CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════

def test_method_convergence() -> Tuple[bool, str, dict]:
    """
    Test that all three pricing methods converge to similar values:
    1. Analytical (Fourier inversion)
    2. PDE (ADI solver)
    3. Monte Carlo (QE scheme)
    
    Expected: All methods within 2% of analytical price
    """
    params = get_default_params()
    K = params.S0
    T = 1.0
    
    # Analytical price (reference)
    pricer = AnalyticalPricer(params)
    price_analytical = pricer.call_price(K, T)
    
    # PDE price
    grid = create_default_grid(params, T, K)
    solver = PDESolver(params, grid)
    U_grid = solver.solve_european_call(K)
    price_pde = solver.get_price(U_grid, params.S0, params.V0)
    
    # Monte Carlo price
    mc = MonteCarloSimulator(params)
    price_mc, stderr = mc.price_european_call(K, T, 252, 100000)
    
    # Compute errors
    error_pde = abs(price_pde - price_analytical) / price_analytical * 100
    error_mc = abs(price_mc - price_analytical) / price_analytical * 100
    
    passed = error_pde < 2.0 and error_mc < 2.0
    message = f"PDE={error_pde:.2f}%, MC={error_mc:.2f}%"
    details = {
        'analytical': price_analytical,
        'pde': price_pde,
        'mc': price_mc,
        'mc_stderr': stderr,
        'error_pde_pct': error_pde,
        'error_mc_pct': error_mc
    }
    
    return passed, message, details


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: GREEKS SIGNS
# ═══════════════════════════════════════════════════════════════════════════════

def test_greeks_signs() -> Tuple[bool, str, dict]:
    """
    Test that Greeks have correct signs for a standard call option:
    
    For a call with S > 0, K > 0, T > 0:
    - Delta > 0 (call increases with stock price)
    - Gamma > 0 (delta increases with stock price)
    - Vega > 0 (call increases with volatility)
    - Theta < 0 (call decreases with time decay)
    - Rho > 0 (call increases with interest rate)
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: FELLER CONDITION
# ═══════════════════════════════════════════════════════════════════════════════

def test_feller_condition() -> Tuple[bool, str, dict]:
    """
    Test Feller condition computation:
    
    Feller Condition: 2κθ > σ²
    
    This ensures the variance process stays strictly positive.
    When satisfied, variance cannot reach zero.
    """
    # Test case that satisfies Feller
    params_ok = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, r=0.05, q=0.02, S0=100.0, V0=0.04)  # 2*2*0.04 = 0.16 > 0.09 = 0.3²
    
    # Test case that violates Feller
    params_bad = HestonParams(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, r=0.05, q=0.02, S0=100.0, V0=0.04)  # 2*1*0.04 = 0.08 < 0.25 = 0.5²
    
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


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: MONTE CARLO VARIANCE POSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def test_mc_variance_positivity() -> Tuple[bool, str, dict]:
    """
    Test that Monte Carlo simulation preserves variance positivity.
    
    The QE (Quadratic-Exponential) scheme is designed to ensure
    V(t) ≥ 0 for all t, even when Feller condition is not satisfied.
    """
    # Use parameters that violate Feller condition
    params = HestonParams(
        kappa=1.0,
        theta=0.04,
        sigma=0.5,  # High vol-of-vol
        rho=-0.7,
        r=0.05,
        q=0.02,
        S0=100.0,
        V0=0.04
    )
    
    mc = MonteCarloSimulator(params)
    S_paths, V_paths = mc.simulate_paths_qe(T=1.0, N_steps=252, N_paths=1000)
    
    # Check all variance values are non-negative
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


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: MONEYNESS EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_moneyness_effects() -> Tuple[bool, str, dict]:
    """
    Test that option prices behave correctly across moneyness:
    
    For call options:
    - ITM (K < S): Higher intrinsic value
    - ATM (K ≈ S): Maximum time value
    - OTM (K > S): Lower price, but positive
    
    The volatility smile (skew) should be visible in implied vols.
    """
    params = get_default_params()
    T = 1.0
    
    pricer = AnalyticalPricer(params)
    bc = BoundaryConditions()
    
    # Test strikes
    strikes = [80, 90, 100, 110, 120]  # ITM to OTM
    prices = []
    impl_vols = []
    
    for K in strikes:
        price = pricer.call_price(K, T)
        iv = implied_volatility_newton(price, params.S0, K, T, params.r, params.q, 'call')
        prices.append(price)
        impl_vols.append(iv)
    
    # Price should decrease as strike increases (for calls)
    prices_decreasing = all(prices[i] > prices[i+1] for i in range(len(prices)-1))
    
    # With negative rho, IV should show downward skew (higher for low strikes)
    # This is the leverage effect
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


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: GRID CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════

def test_grid_convergence() -> Tuple[bool, str, dict]:
    """
    Test PDE solver convergence as grid is refined.
    
    As N_S, N_V, N_t increase, PDE price should converge to analytical.
    Error should decrease with refinement.
    """
    params = get_default_params()
    K = params.S0
    T = 1.0
    
    # Analytical reference
    pricer = AnalyticalPricer(params)
    price_ref = pricer.call_price(K, T)
    
    # Coarse grid - use create_default_grid for convenience
    grid_coarse = create_default_grid(params, T, K)
    solver_coarse = PDESolver(params, grid_coarse)
    U_coarse = solver_coarse.solve_european_call(K)
    price_coarse = solver_coarse.get_price(U_coarse, params.S0, params.V0)
    
    # Fine grid - create with more points
    S_min, S_max = params.S0 * 0.2, params.S0 * 3.0
    V_min, V_max = 1e-8, max(params.V0, params.theta) * 5
    grid_fine = Grid(S_min, S_max, 100, V_min, V_max, 50, T, 100)
    solver_fine = PDESolver(params, grid_fine)
    U_fine = solver_fine.solve_european_call(K)
    price_fine = solver_fine.get_price(U_fine, params.S0, params.V0)
    
    error_coarse = abs(price_coarse - price_ref)
    error_fine = abs(price_fine - price_ref)
    
    # Fine grid should have smaller error
    convergent = error_fine < error_coarse
    
    passed = convergent and error_fine < 0.1
    message = f"Coarse error={error_coarse:.4f}, Fine error={error_fine:.4f}"
    details = {
        'reference': price_ref,
        'coarse': price_coarse,
        'fine': price_fine,
        'error_coarse': error_coarse,
        'error_fine': error_fine,
        'convergent': convergent
    }
    
    return passed, message, details


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests() -> bool:
    """Run all validation tests and report results."""
    
    print("=" * 70)
    print("HESTON ENGINE VALIDATION SUITE")
    print("=" * 70)
    print()
    
    tests = [
        ("Put-Call Parity", test_put_call_parity),
        ("Black-Scholes Limit", test_bs_limit),
        ("Method Convergence", test_method_convergence),
        ("Greeks Signs", test_greeks_signs),
        ("Feller Condition", test_feller_condition),
        ("MC Variance Positivity", test_mc_variance_positivity),
        ("Moneyness Effects", test_moneyness_effects),
        ("Grid Convergence", test_grid_convergence),
    ]
    
    results = []
    for name, test_func in tests:
        result = run_test(name, test_func)
        results.append(result)
        print(result)
        if result.details:
            for key, value in result.details.items():
                if key != 'checks' and not isinstance(value, (list, dict)):
                    print(f"    {key}: {value}")
        print()
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print("=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n✓ All validation tests PASSED!")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) FAILED")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
