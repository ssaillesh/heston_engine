from typing import Tuple, Dict

try:
    from .common import get_default_params, AnalyticalPricer, create_default_grid, PDESolver, Grid, is_numerically_stable
except ImportError:
    from common import get_default_params, AnalyticalPricer, create_default_grid, PDESolver, Grid, is_numerically_stable


def test_grid_convergence() -> Tuple[bool, str, Dict]:
    params = get_default_params()
    K = params.S0
    T = 1.0

    pricer = AnalyticalPricer(params)
    price_ref = pricer.call_price(K, T)

    grid_coarse = create_default_grid(params, T, K)
    solver_coarse = PDESolver(params, grid_coarse)
    U_coarse = solver_coarse.solve_european_call(K)
    price_coarse = solver_coarse.get_price(U_coarse, params.S0, params.V0)
    coarse_fallback_used = solver_coarse.last_solve_used_fallback

    error_coarse = abs(price_coarse - price_ref)
    coarse_stable = is_numerically_stable(price_coarse)

    S_min, S_max = params.S0 * 0.3, params.S0 * 2.5
    V_min, V_max = 1e-8, max(0.5, max(params.V0, params.theta) * 5)
    grid_fine = Grid(S_min, S_max, 110, V_min, V_max, 55, T, 400)
    solver_fine = PDESolver(params, grid_fine)
    U_fine = solver_fine.solve_european_call(K)
    price_fine = solver_fine.get_price(U_fine, params.S0, params.V0)
    fine_fallback_used = solver_fine.last_solve_used_fallback

    error_fine = abs(price_fine - price_ref)
    fine_stable = is_numerically_stable(price_fine)

    convergent = error_fine < error_coarse

    passed = (
        coarse_stable and
        fine_stable and
        (not coarse_fallback_used) and
        (not fine_fallback_used) and
        convergent and
        error_fine < 0.1
    )
    message = f"Coarse error={error_coarse:.4f}, Fine error={error_fine:.4f}"
    details = {
        'reference': price_ref,
        'coarse': price_coarse,
        'fine': price_fine,
        'error_coarse': error_coarse,
        'error_fine': error_fine,
        'coarse_stable': coarse_stable,
        'fine_stable': fine_stable,
        'coarse_fallback_used': coarse_fallback_used,
        'fine_fallback_used': fine_fallback_used,
        'convergent': convergent
    }

    return passed, message, details
