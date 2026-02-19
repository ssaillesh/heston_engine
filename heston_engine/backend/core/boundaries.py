"""
Boundary Conditions for European Options under Heston Model

═══════════════════════════════════════════════════════════════════════════════
BOUNDARY CONDITIONS FOR HESTON PDE
═══════════════════════════════════════════════════════════════════════════════

The Heston PDE must be supplemented with boundary conditions on all edges
of the computational domain [S_min, S_max] × [V_min, V_max] × [0, T].

1. TERMINAL CONDITION (τ = 0, i.e., t = T):
   ═══════════════════════════════════════════════════════════════════════════
   
   At expiry, option value equals payoff:
   
   European Call: V(S, V, T) = max(S - K, 0) = (S - K)⁺
   European Put:  V(S, V, T) = max(K - S, 0) = (K - S)⁺
   
   This is the starting point for backward time-stepping.

2. S = 0 BOUNDARY (Asset price falls to zero):
   ═══════════════════════════════════════════════════════════════════════════
   
   If S = 0, it stays at 0 (absorbing state for geometric Brownian motion).
   
   Call option: V(0, V, t) = 0
     (Call worthless if S = 0)
   
   Put option: V(0, V, t) = K·e^{-r(T-t)}
     (Put worth discounted strike)

3. S → ∞ BOUNDARY:
   ═══════════════════════════════════════════════════════════════════════════
   
   As S → ∞, option behavior approaches limiting case:
   
   Call: V(S, V, t) → S·e^{-q(T-t)} - K·e^{-r(T-t)}
     (Deep ITM call = forward price minus discounted strike)
   
   Put: V(S, V, t) → 0
     (Deep ITM put becomes worthless)
   
   Implementation: Often use linear extrapolation from interior.

4. V = 0 BOUNDARY (Variance hits zero):
   ═══════════════════════════════════════════════════════════════════════════
   
   When V = 0, the Heston model degenerates:
   
   dS = (r - q)S dt  (no diffusion)
   dV = κθ dt        (drift only, pushes V positive)
   
   For PDE: This is an "outflow" boundary if Feller satisfied.
   
   Approach 1: Let PDE determine values (natural boundary)
   Approach 2: Use Black-Scholes with σ = √θ (limiting volatility)
   
   Black-Scholes approximation:
   V(S, 0, t) ≈ BS_call(S, K, T-t, r, q, √θ)

5. V → ∞ BOUNDARY:
   ═══════════════════════════════════════════════════════════════════════════
   
   As V → ∞, option value has asymptotic behavior:
   
   Call: V(S, V, t) → S·e^{-q(T-t)}
     (Extreme volatility → call worth discounted spot)
   
   Put: V(S, V, t) → K·e^{-r(T-t)}
     (Extreme volatility → put worth discounted strike)
   
   Implementation: Set as Dirichlet or use Neumann (∂V/∂V = 0).

═══════════════════════════════════════════════════════════════════════════════
BLACK-SCHOLES FORMULA (for V=0 boundary)
═══════════════════════════════════════════════════════════════════════════════

C(S, K, τ, r, q, σ) = S·e^{-qτ}·N(d₁) - K·e^{-rτ}·N(d₂)

where:
d₁ = [ln(S/K) + (r - q + σ²/2)τ] / (σ√τ)
d₂ = d₁ - σ√τ

N(·) = standard normal CDF

Put-Call Parity:
P = C - S·e^{-qτ} + K·e^{-rτ}

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.stats import norm
from typing import Callable, Optional


class BoundaryConditions:
    """
    Boundary condition handlers for European options under Heston model.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TERMINAL CONDITIONS (PAYOFF FUNCTIONS)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def call_payoff(S: np.ndarray, K: float) -> np.ndarray:
        """
        European call option payoff at expiry.
        
        Formula:
        Payoff(S) = max(S - K, 0) = (S - K)⁺
        
        This is the terminal condition V(S, V, T) for call options.
        At expiry, the option is worth its intrinsic value.
        
        Args:
            S: Asset price(s) - scalar or array
            K: Strike price
            
        Returns:
            Payoff value(s)
        """
        return np.maximum(S - K, 0.0)
    
    @staticmethod
    def put_payoff(S: np.ndarray, K: float) -> np.ndarray:
        """
        European put option payoff at expiry.
        
        Formula:
        Payoff(S) = max(K - S, 0) = (K - S)⁺
        
        This is the terminal condition V(S, V, T) for put options.
        
        Args:
            S: Asset price(s) - scalar or array
            K: Strike price
            
        Returns:
            Payoff value(s)
        """
        return np.maximum(K - S, 0.0)
    
    @staticmethod
    def digital_call_payoff(S: np.ndarray, K: float) -> np.ndarray:
        """
        Digital (binary) call payoff.
        
        Formula:
        Payoff(S) = 1 if S ≥ K, else 0
        
        Args:
            S: Asset price(s)
            K: Strike price
            
        Returns:
            Payoff value(s) (0 or 1)
        """
        return np.where(S >= K, 1.0, 0.0)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BLACK-SCHOLES FORMULA (for V=0 boundary and validation)
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def black_scholes_call(
        S: float, 
        K: float, 
        tau: float, 
        r: float, 
        q: float, 
        sigma: float
    ) -> float:
        """
        Black-Scholes European call option formula.
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        C = S·e^{-qτ}·N(d₁) - K·e^{-rτ}·N(d₂)
        
        where:
        d₁ = [ln(S/K) + (r - q + σ²/2)τ] / (σ√τ)
        d₂ = d₁ - σ√τ
        
        N(x) = (1/√(2π)) ∫_{-∞}^{x} e^{-t²/2} dt  (standard normal CDF)
        
        Derivation:
        - Start from risk-neutral pricing: C = e^{-rτ}·E^Q[(S_T - K)⁺]
        - S_T = S·exp((r-q-σ²/2)τ + σ√τ·Z) where Z ~ N(0,1)
        - Work out expectation using properties of lognormal distribution
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            S: Current spot price
            K: Strike price  
            tau: Time to maturity (T - t)
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility (constant)
            
        Returns:
            Call option price
        """
        # Handle edge cases
        if tau < 1e-10:
            # At expiry, return payoff
            return max(S - K, 0.0)
        
        if sigma < 1e-10:
            # Zero volatility case: deterministic forward
            forward = S * np.exp((r - q) * tau)
            return max(forward - K, 0.0) * np.exp(-r * tau)
        
        if S < 1e-10:
            # S = 0: call worthless
            return 0.0
        
        # Compute d1 and d2
        # d₁ = [ln(S/K) + (r - q + σ²/2)τ] / (σ√τ)
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        
        # d₂ = d₁ - σ√τ
        d2 = d1 - sigma * sqrt_tau
        
        # C = S·e^{-qτ}·N(d₁) - K·e^{-rτ}·N(d₂)
        call_price = (S * np.exp(-q * tau) * norm.cdf(d1) - 
                     K * np.exp(-r * tau) * norm.cdf(d2))
        
        return call_price
    
    @staticmethod
    def black_scholes_put(
        S: float, 
        K: float, 
        tau: float, 
        r: float, 
        q: float, 
        sigma: float
    ) -> float:
        """
        Black-Scholes European put option formula.
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        P = K·e^{-rτ}·N(-d₂) - S·e^{-qτ}·N(-d₁)
        
        Alternatively, via put-call parity:
        P = C - S·e^{-qτ} + K·e^{-rτ}
        
        where C is the call price with same parameters.
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            S: Current spot price
            K: Strike price
            tau: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
            
        Returns:
            Put option price
        """
        if tau < 1e-10:
            return max(K - S, 0.0)
        
        if sigma < 1e-10:
            forward = S * np.exp((r - q) * tau)
            return max(K - forward, 0.0) * np.exp(-r * tau)
        
        if S < 1e-10:
            return K * np.exp(-r * tau)
        
        # Use put-call parity: P = C - S·e^{-qτ} + K·e^{-rτ}
        call_price = BoundaryConditions.black_scholes_call(S, K, tau, r, q, sigma)
        put_price = call_price - S * np.exp(-q * tau) + K * np.exp(-r * tau)
        
        return put_price
    
    @staticmethod
    def black_scholes_delta(
        S: float,
        K: float, 
        tau: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Black-Scholes Delta (∂V/∂S).
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        Call Delta: Δ_C = e^{-qτ}·N(d₁)
        Put Delta:  Δ_P = -e^{-qτ}·N(-d₁) = e^{-qτ}·[N(d₁) - 1]
        
        Delta interpretation:
        - Number of shares to hold for delta-hedging
        - Probability proxy for ending ITM (risk-neutral)
        ═══════════════════════════════════════════════════════════════════════
        """
        if tau < 1e-10 or sigma < 1e-10 or S < 1e-10:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        
        if option_type == 'call':
            return np.exp(-q * tau) * norm.cdf(d1)
        else:
            return np.exp(-q * tau) * (norm.cdf(d1) - 1)
    
    @staticmethod
    def black_scholes_gamma(
        S: float,
        K: float,
        tau: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """
        Black-Scholes Gamma (∂²V/∂S²).
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        Γ = e^{-qτ}·n(d₁) / (S·σ·√τ)
        
        where n(x) = (1/√(2π))·e^{-x²/2} is the standard normal PDF.
        
        Gamma is the same for calls and puts.
        ═══════════════════════════════════════════════════════════════════════
        """
        if tau < 1e-10 or sigma < 1e-10 or S < 1e-10:
            return 0.0
        
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        
        return np.exp(-q * tau) * norm.pdf(d1) / (S * sigma * sqrt_tau)
    
    @staticmethod
    def black_scholes_vega(
        S: float,
        K: float,
        tau: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """
        Black-Scholes Vega (∂V/∂σ).
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        ν = S·e^{-qτ}·√τ·n(d₁)
        
        Vega is the same for calls and puts.
        Often quoted per 1% move in volatility (divide by 100).
        ═══════════════════════════════════════════════════════════════════════
        """
        if tau < 1e-10 or sigma < 1e-10 or S < 1e-10:
            return 0.0
        
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        
        return S * np.exp(-q * tau) * sqrt_tau * norm.pdf(d1)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPATIAL BOUNDARY CONDITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def apply_S_min_boundary(
        V_grid: np.ndarray,
        K: float,
        tau: float,
        r: float,
        option_type: str = 'call'
    ) -> np.ndarray:
        """
        Apply S = S_min boundary condition.
        
        Boundary Condition at S = 0:
        ═══════════════════════════════════════════════════════════════════════
        
        Call: V(0, V, t) = 0
          When S = 0, it stays at 0 (geometric BM is absorbing at 0)
          Call payoff (S - K)⁺ at S = 0 is always 0
        
        Put: V(0, V, t) = K·e^{-r(T-t)}
          When S = 0, put is certain to pay K at expiry
          Value today is discounted strike
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            V_grid: Solution grid, shape (N_S, N_V)
            K: Strike price
            tau: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            
        Returns:
            Modified V_grid with boundary applied at S_min (index 0)
        """
        if option_type == 'call':
            V_grid[0, :] = 0.0
        else:
            V_grid[0, :] = K * np.exp(-r * tau)
        
        return V_grid
    
    @staticmethod
    def apply_S_max_boundary(
        V_grid: np.ndarray,
        S_max: float,
        K: float,
        tau: float,
        r: float,
        q: float,
        option_type: str = 'call'
    ) -> np.ndarray:
        """
        Apply S = S_max boundary condition.
        
        Boundary Condition at S → ∞:
        ═══════════════════════════════════════════════════════════════════════
        
        Call: V(∞, V, t) → S·e^{-qτ} - K·e^{-rτ}
          Deep ITM call behaves like forward minus discounted strike
          Delta → 1 as S → ∞
        
        Put: V(∞, V, t) → 0
          Deep OTM put becomes worthless
          Delta → 0 as S → ∞
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            V_grid: Solution grid
            S_max: Maximum S value
            K: Strike price
            tau: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            option_type: 'call' or 'put'
            
        Returns:
            Modified V_grid with boundary applied at S_max
        """
        if option_type == 'call':
            # V(S_max) ≈ S_max·e^{-qτ} - K·e^{-rτ}
            V_grid[-1, :] = max(S_max * np.exp(-q * tau) - K * np.exp(-r * tau), 0.0)
        else:
            # V(S_max) ≈ 0
            V_grid[-1, :] = 0.0
        
        return V_grid
    
    @staticmethod
    def apply_V_min_boundary(
        V_grid: np.ndarray,
        S_grid: np.ndarray,
        K: float,
        tau: float,
        r: float,
        q: float,
        theta: float,
        option_type: str = 'call'
    ) -> np.ndarray:
        """
        Apply V = V_min boundary condition.
        
        Boundary Condition at V = 0:
        ═══════════════════════════════════════════════════════════════════════
        
        When variance hits zero, the model degenerates to:
        dS = (r - q)S dt  (deterministic growth)
        dV = κθ dt        (variance drifts positive)
        
        Options approach Black-Scholes limit with long-term volatility:
        V(S, 0, t) ≈ BS(S, K, τ, r, q, √θ)
        
        Rationale: As V → 0, future variance will revert to θ,
        so √θ is a reasonable effective volatility.
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            V_grid: Solution grid
            S_grid: S values along first dimension
            K: Strike price
            tau: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            theta: Long-term variance
            option_type: 'call' or 'put'
            
        Returns:
            Modified V_grid with boundary applied at V_min
        """
        sigma_eff = np.sqrt(theta)  # Effective volatility = √θ
        
        for i, S in enumerate(S_grid):
            if option_type == 'call':
                V_grid[i, 0] = BoundaryConditions.black_scholes_call(
                    S, K, tau, r, q, sigma_eff
                )
            else:
                V_grid[i, 0] = BoundaryConditions.black_scholes_put(
                    S, K, tau, r, q, sigma_eff
                )
        
        return V_grid
    
    @staticmethod
    def apply_V_max_boundary(
        V_grid: np.ndarray,
        S_grid: np.ndarray,
        tau: float,
        q: float,
        r: float,
        K: float,
        option_type: str = 'call'
    ) -> np.ndarray:
        """
        Apply V = V_max boundary condition.
        
        Boundary Condition at V → ∞:
        ═══════════════════════════════════════════════════════════════════════
        
        As variance becomes very large:
        
        Call: V(S, ∞, t) → S·e^{-qτ}
          Extreme volatility → call converges to discounted asset value
        
        Put: V(S, ∞, t) → K·e^{-rτ}
          Extreme volatility → put converges to discounted strike
        
        Mathematical reason: With infinite volatility, both outcomes
        (S_T >> K and S_T << K) are equally likely in some sense,
        leading to these asymptotic values.
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            V_grid: Solution grid
            S_grid: S values along first dimension
            tau: Time to maturity
            q: Dividend yield
            r: Risk-free rate
            K: Strike price
            option_type: 'call' or 'put'
            
        Returns:
            Modified V_grid with boundary applied at V_max
        """
        if option_type == 'call':
            # V(S, V_max) ≈ S·e^{-qτ}
            V_grid[:, -1] = S_grid * np.exp(-q * tau)
        else:
            # V(S, V_max) ≈ K·e^{-rτ}
            V_grid[:, -1] = K * np.exp(-r * tau)
        
        return V_grid


def implied_volatility_newton(
    price: float,
    S: float,
    K: float,
    tau: float,
    r: float,
    q: float,
    option_type: str = 'call',
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Compute implied volatility using Newton-Raphson method.
    
    Newton-Raphson for Implied Volatility:
    ═══════════════════════════════════════════════════════════════════════════
    
    Find σ such that BS(S, K, τ, r, q, σ) = market_price
    
    Iteration:
    σ_{n+1} = σ_n - [BS(σ_n) - price] / Vega(σ_n)
    
    Convergence: Quadratic near solution, may fail far from solution
    
    Initial guess: σ₀ = √(2|ln(S/K)|/τ) (Brenner-Subrahmanyam)
    
    ═══════════════════════════════════════════════════════════════════════════
    
    Args:
        price: Market option price
        S: Spot price
        K: Strike
        tau: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        option_type: 'call' or 'put'
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Implied volatility
    """
    bc = BoundaryConditions()
    
    # Initial guess (Brenner-Subrahmanyam approximation)
    # σ ≈ √(2π/T) · C/S for ATM options
    # More general: σ₀ = √(2|ln(S/K)|/τ)
    sigma = np.sqrt(2 * np.abs(np.log(S / K)) / max(tau, 0.01)) + 0.2
    sigma = max(0.01, min(sigma, 3.0))  # Bound initial guess
    
    for _ in range(max_iter):
        # Compute BS price and vega
        if option_type == 'call':
            bs_price = bc.black_scholes_call(S, K, tau, r, q, sigma)
        else:
            bs_price = bc.black_scholes_put(S, K, tau, r, q, sigma)
        
        vega = bc.black_scholes_vega(S, K, tau, r, q, sigma)
        
        # Check convergence
        error = bs_price - price
        if abs(error) < tol:
            return sigma
        
        # Newton update: σ_{n+1} = σ_n - error / vega
        if abs(vega) < 1e-10:
            # Vega too small, use bisection fallback
            break
        
        sigma_new = sigma - error / vega
        sigma = max(0.01, min(sigma_new, 5.0))  # Keep bounded
    
    # Fallback to bisection if Newton fails
    return implied_volatility_bisection(price, S, K, tau, r, q, option_type)


def implied_volatility_bisection(
    price: float,
    S: float,
    K: float,
    tau: float,
    r: float,
    q: float,
    option_type: str = 'call',
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Compute implied volatility using bisection method.
    
    Bisection Method:
    ═══════════════════════════════════════════════════════════════════════════
    
    Find σ in [σ_low, σ_high] such that BS(σ) = price
    
    Algorithm:
    1. Start with bracket [0.01, 5.0]
    2. σ_mid = (σ_low + σ_high) / 2
    3. If BS(σ_mid) > price: σ_high = σ_mid
    4. Else: σ_low = σ_mid
    5. Repeat until |BS(σ_mid) - price| < tol
    
    Guaranteed convergence (linear rate) if solution exists in bracket.
    
    ═══════════════════════════════════════════════════════════════════════════
    
    Args:
        price: Market option price
        S: Spot price
        K: Strike
        tau: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        option_type: 'call' or 'put'
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Implied volatility (or NaN if not found)
    """
    bc = BoundaryConditions()
    
    sigma_low = 0.001
    sigma_high = 5.0
    
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        
        if option_type == 'call':
            bs_price = bc.black_scholes_call(S, K, tau, r, q, sigma_mid)
        else:
            bs_price = bc.black_scholes_put(S, K, tau, r, q, sigma_mid)
        
        error = bs_price - price
        
        if abs(error) < tol:
            return sigma_mid
        
        if error > 0:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    
    return np.nan  # Failed to converge
