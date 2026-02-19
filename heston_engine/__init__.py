"""
═══════════════════════════════════════════════════════════════════════════════
HESTON ENGINE - Stochastic Volatility Option Pricing
═══════════════════════════════════════════════════════════════════════════════

A complete implementation of the Heston (1993) stochastic volatility model
for pricing European options.

Mathematical Model:
    dS = (r-q)S dt + √V S dW_S
    dV = κ(θ-V)dt + σ√V dW_V
    Corr(dW_S, dW_V) = ρ

Where:
    S  = Stock price
    V  = Instantaneous variance
    r  = Risk-free rate
    q  = Dividend yield
    κ  = Mean reversion speed
    θ  = Long-term variance
    σ  = Volatility of variance (vol of vol)
    ρ  = Correlation between stock and variance

Modules:
    backend.core        - Core classes (parameters, grid, boundaries)
    backend.solvers     - Pricing engines (PDE, analytical, Monte Carlo)
    backend.calibration - Model calibration to market data
    backend.greeks      - Greeks/sensitivities computation
    tests               - Validation tests

Usage:
    from heston_engine.backend.core.parameters import HestonParams, get_default_params
    from heston_engine.backend.solvers.analytical import AnalyticalPricer
    
    params = get_default_params()
    pricer = AnalyticalPricer(params)
    price = pricer.call_price(K=100, T=1.0)

═══════════════════════════════════════════════════════════════════════════════
"""

__version__ = '1.0.0'
__author__ = 'Heston Engine'

from backend.core.parameters import HestonParams, get_default_params
from backend.solvers.analytical import AnalyticalPricer
from backend.solvers.monte_carlo import MonteCarloSimulator
from backend.solvers.pde_solver import PDESolver
