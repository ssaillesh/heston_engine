#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
HESTON ENGINE - Application Entry Point
═══════════════════════════════════════════════════════════════════════════════

This script starts the Heston Stochastic Volatility Model Engine.

Usage:
    python run.py              # Start web server
    python run.py --test       # Run validation tests only
    python run.py --demo       # Run demo pricing calculations

Mathematical Model:
    dS = (r-q)S dt + √V S dW_S
    dV = κ(θ-V)dt + σ√V dW_V
    Corr(dW_S, dW_V) = ρ

Parameters:
    κ (kappa)  - Mean reversion speed
    θ (theta)  - Long-term variance
    σ (sigma)  - Volatility of variance (vol of vol)
    ρ (rho)    - Correlation between S and V
    r          - Risk-free interest rate
    q          - Dividend yield

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import argparse


def run_demo():
    """Run demonstration calculations."""
    
    print("=" * 70)
    print("HESTON ENGINE - DEMO")
    print("=" * 70)
    print()
    
    from heston_engine.backend.core.parameters import get_default_params
    from heston_engine.backend.solvers.analytical import AnalyticalPricer
    from heston_engine.backend.solvers.monte_carlo import MonteCarloSimulator
    from heston_engine.backend.greeks.calculator import GreeksCalculator
    
    # Default parameters
    params = get_default_params()
    print("Heston Parameters:")
    print(f"  κ (kappa)  = {params.kappa}")
    print(f"  θ (theta)  = {params.theta}")
    print(f"  σ (sigma)  = {params.sigma}")
    print(f"  ρ (rho)    = {params.rho}")
    print(f"  r          = {params.r}")
    print(f"  q          = {params.q}")
    print(f"  S₀         = {params.S0}")
    print(f"  V₀         = {params.V0}")
    print(f"  Feller     = {params.feller_ratio:.2f} {'✓' if params.feller_satisfied else '✗'}")
    print()
    
    # Option parameters
    K = 100.0  # Strike
    T = 1.0    # Maturity (1 year)
    
    print(f"Option: European Call, K={K}, T={T}")
    print("-" * 70)
    
    # Analytical pricing
    print("\n1. ANALYTICAL PRICING (Fourier Inversion)")
    pricer = AnalyticalPricer(params)
    call_price = pricer.call_price(K, T)
    put_price = pricer.put_price(K, T)
    print(f"   Call Price: {call_price:.4f}")
    print(f"   Put Price:  {put_price:.4f}")
    
    # Monte Carlo
    print("\n2. MONTE CARLO (QE Scheme, 100,000 paths)")
    mc = MonteCarloSimulator(params)
    mc_price, mc_stderr = mc.price_european_call(K, T, 252, 100000)
    print(f"   Call Price: {mc_price:.4f} ± {mc_stderr:.4f}")
    print(f"   95% CI: [{mc_price - 1.96*mc_stderr:.4f}, {mc_price + 1.96*mc_stderr:.4f}]")
    
    # Greeks
    print("\n3. GREEKS")
    calc = GreeksCalculator(params)
    greeks = calc.all_greeks(K, T, 'call')
    print(f"   Delta (Δ): {greeks['delta']:.4f}")
    print(f"   Gamma (Γ): {greeks['gamma']:.6f}")
    print(f"   Vega  (ν): {greeks['vega']:.4f}")
    print(f"   Theta (Θ): {greeks['theta']:.4f} (daily: {greeks['theta']/365:.4f})")
    print(f"   Rho   (ρ): {greeks['rho']:.4f}")
    print(f"   Vanna:     {greeks['vanna']:.6f}")
    print(f"   Volga:     {greeks['volga']:.4f}")
    
    # Implied volatility surface
    from heston_engine.backend.core.boundaries import implied_volatility_newton
    print("\n4. IMPLIED VOLATILITY SMILE")
    strikes = [85, 90, 95, 100, 105, 110, 115]
    print("   Strike    Price    Implied Vol")
    for strike in strikes:
        price = pricer.call_price(strike, T)
        iv = implied_volatility_newton(price, params.S0, strike, T, params.r, params.q, 'call')
        print(f"   {strike:6.0f}    {price:6.2f}    {iv*100:5.2f}%")
    
    print()
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


def run_tests():
    """Run validation tests."""
    from heston_engine.tests.validation import run_all_tests
    success = run_all_tests()
    return 0 if success else 1


def run_server(host='0.0.0.0', port=5000, debug=True):
    """Start the web server."""
    from heston_engine.backend.app import app
    app.run(host=host, port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser(
        description='Heston Stochastic Volatility Model Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py              Start web server at http://localhost:5000
    python run.py --port 8000  Start on custom port
    python run.py --test       Run validation tests
    python run.py --demo       Run demo calculations
        """
    )
    
    parser.add_argument('--test', action='store_true', help='Run validation tests')
    parser.add_argument('--demo', action='store_true', help='Run demo calculations')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')
    
    args = parser.parse_args()
    
    if args.test:
        sys.exit(run_tests())
    elif args.demo:
        run_demo()
    else:
        run_server(args.host, args.port, not args.no_debug)


if __name__ == '__main__':
    main()
