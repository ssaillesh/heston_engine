from typing import Tuple, Dict

try:
    from .common import get_default_params, AnalyticalPricer, create_default_grid, PDESolver, MonteCarloSimulator, is_numerically_stable
except ImportError:
    from common import get_default_params, AnalyticalPricer, create_default_grid, PDESolver, MonteCarloSimulator, is_numerically_stable


def test_method_convergence() -> Tuple[bool, str, Dict]:
    params = get_default_params()
    K = params.S0
    T = 1.0

    pricer = AnalyticalPricer(params)
    price_analytical = pricer.call_price(K, T)

    grid = create_default_grid(params, T, K)
    solver = PDESolver(params, grid)
    U_grid = solver.solve_european_call(K)
    price_pde = solver.get_price(U_grid, params.S0, params.V0)
    pde_fallback_used = solver.last_solve_used_fallback

    mc = MonteCarloSimulator(params)
    price_mc, stderr = mc.price_european_call(K, T, 252, 50000, seed=42)

    pde_stable = is_numerically_stable(price_pde)
    error_pde = abs(price_pde - price_analytical) / price_analytical * 100 if pde_stable else float('inf')
    error_mc = abs(price_mc - price_analytical) / price_analytical * 100

    passed = pde_stable and (not pde_fallback_used) and error_pde < 2.0 and error_mc < 2.0
    message = f"PDE={error_pde:.2f}%, MC={error_mc:.2f}%, fallback={pde_fallback_used}"

    details = {
        'analytical': price_analytical,
        'pde': price_pde,
        'pde_stable': pde_stable,
        'pde_fallback_used': pde_fallback_used,
        'mc': price_mc,
        'mc_stderr': stderr,
        'error_pde_pct': error_pde,
        'error_mc_pct': error_mc
    }

    return passed, message, details
