"""
Market Data Handlers and Implied Volatility Calculations

═══════════════════════════════════════════════════════════════════════════════
MARKET DATA PROCESSING FOR CALIBRATION
═══════════════════════════════════════════════════════════════════════════════

1. IMPLIED VOLATILITY CALCULATION:
   ═══════════════════════════════════════════════════════════════════════════
   
   Given market price C_market, find σ_IV such that:
   BS(S, K, T, r, q, σ_IV) = C_market
   
   Newton-Raphson iteration:
   σ_{n+1} = σ_n - [BS(σ_n) - C_market] / Vega(σ_n)
   
   Convergence typically in 3-5 iterations.

2. VOLATILITY SURFACE:
   ═══════════════════════════════════════════════════════════════════════════
   
   Market quotes organized by:
   - Strike (K) or moneyness (K/S, log(K/F))
   - Maturity (T) or tenor
   
   Surface interpolation methods:
   - Bilinear in (K, T)
   - SABR parameterization per maturity
   - Spline interpolation

3. SMILE CHARACTERISTICS:
   ═══════════════════════════════════════════════════════════════════════════
   
   Volatility skew: ∂σ_IV/∂K < 0 typically (equity markets)
   Volatility term structure: ∂σ_IV/∂T can be positive or negative
   
   Smile curvature (convexity): ∂²σ_IV/∂K² > 0
   
   Risk reversals: σ_IV(25Δ put) - σ_IV(25Δ call)
   Butterflies: 0.5[σ_IV(25Δ put) + σ_IV(25Δ call)] - σ_IV(ATM)

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VolQuote:
    """Single volatility quote."""
    strike: float
    maturity: float
    implied_vol: float
    option_type: str = 'call'
    delta: Optional[float] = None
    bid_vol: Optional[float] = None
    ask_vol: Optional[float] = None


class ImpliedVolCalculator:
    """
    Calculate implied volatility from option prices.
    """
    
    @staticmethod
    def black_scholes_call(S, K, T, r, q, sigma):
        """
        Black-Scholes call price.
        
        C = S·e^{-qT}·N(d₁) - K·e^{-rT}·N(d₂)
        """
        if T < 1e-10:
            return max(S - K, 0)
        if sigma < 1e-10:
            return max(S*np.exp(-q*T) - K*np.exp(-r*T), 0)
        
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt_T)
        d2 = d1 - sigma*sqrt_T
        
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    @staticmethod
    def black_scholes_vega(S, K, T, r, q, sigma):
        """
        Black-Scholes vega.
        
        ν = S·e^{-qT}·√T·n(d₁)
        """
        if T < 1e-10 or sigma < 1e-10:
            return 0.0
        
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt_T)
        
        return S*np.exp(-q*T)*sqrt_T*norm.pdf(d1)
    
    def implied_vol_newton(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: str = 'call',
        tol: float = 1e-8,
        max_iter: int = 100
    ) -> float:
        """
        Implied volatility via Newton-Raphson.
        
        Newton-Raphson Algorithm:
        ═══════════════════════════════════════════════════════════════════════
        
        σ_{n+1} = σ_n - f(σ_n)/f'(σ_n)
        
        where:
        f(σ) = BS(σ) - C_market
        f'(σ) = Vega(σ)
        
        Initial guess: Brenner-Subrahmanyam approximation
        σ₀ = √(2π/T) · C/(S·e^{-qT})  for ATM options
        
        ═══════════════════════════════════════════════════════════════════════
        """
        # Adjust for puts using put-call parity
        if option_type == 'put':
            # C = P + S·e^{-qT} - K·e^{-rT}
            price = price + S*np.exp(-q*T) - K*np.exp(-r*T)
        
        # Initial guess
        sigma = np.sqrt(2 * np.abs(np.log(S/K)) / max(T, 0.01)) + 0.2
        sigma = max(0.01, min(sigma, 3.0))
        
        for _ in range(max_iter):
            bs_price = self.black_scholes_call(S, K, T, r, q, sigma)
            vega = self.black_scholes_vega(S, K, T, r, q, sigma)
            
            error = bs_price - price
            
            if abs(error) < tol:
                return sigma
            
            if abs(vega) < 1e-10:
                break  # Fall back to bisection
            
            sigma_new = sigma - error / vega
            sigma = max(0.001, min(sigma_new, 5.0))
        
        # Fallback to bisection
        return self.implied_vol_bisection(price, S, K, T, r, q, option_type)
    
    def implied_vol_bisection(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: str = 'call',
        tol: float = 1e-8
    ) -> float:
        """
        Implied volatility via bisection.
        
        Guaranteed to converge if solution exists in [σ_min, σ_max].
        """
        if option_type == 'put':
            price = price + S*np.exp(-q*T) - K*np.exp(-r*T)
        
        def objective(sigma):
            return self.black_scholes_call(S, K, T, r, q, sigma) - price
        
        try:
            result: float = brentq(objective, 0.001, 5.0, xtol=tol)  # type: ignore[assignment]
            return result
        except:
            return float(np.nan)


class VolatilitySurface:
    """
    Volatility surface container and interpolator.
    """
    
    def __init__(self, quotes: List[VolQuote]):
        """
        Initialize volatility surface from quotes.
        
        Args:
            quotes: List of VolQuote objects
        """
        self.quotes = quotes
        
        # Extract unique strikes and maturities
        self.strikes = sorted(set(q.strike for q in quotes))
        self.maturities = sorted(set(q.maturity for q in quotes))
        
        # Build surface grid
        self._build_grid()
    
    def _build_grid(self):
        """Build interpolation grid."""
        n_K = len(self.strikes)
        n_T = len(self.maturities)
        
        self.grid = np.full((n_T, n_K), np.nan)
        
        strike_idx = {K: i for i, K in enumerate(self.strikes)}
        mat_idx = {T: i for i, T in enumerate(self.maturities)}
        
        for q in self.quotes:
            i = mat_idx.get(q.maturity)
            j = strike_idx.get(q.strike)
            if i is not None and j is not None:
                self.grid[i, j] = q.implied_vol
    
    def interpolate(self, K: float, T: float) -> float:
        """
        Interpolate implied volatility at (K, T).
        
        Bilinear Interpolation:
        ═══════════════════════════════════════════════════════════════════════
        
        σ(K, T) = (1-α)(1-β)σ₀₀ + α(1-β)σ₁₀ + (1-α)β·σ₀₁ + αβ·σ₁₁
        
        where:
        α = (K - K_i)/(K_{i+1} - K_i)
        β = (T - T_j)/(T_{j+1} - T_j)
        
        ═══════════════════════════════════════════════════════════════════════
        """
        # Find bracketing indices
        i_K = np.searchsorted(self.strikes, K) - 1
        i_K = max(0, min(i_K, len(self.strikes) - 2))
        
        i_T = np.searchsorted(self.maturities, T) - 1
        i_T = max(0, min(i_T, len(self.maturities) - 2))
        
        # Interpolation weights
        K0, K1 = self.strikes[i_K], self.strikes[i_K + 1]
        T0, T1 = self.maturities[i_T], self.maturities[i_T + 1]
        
        alpha = (K - K0) / (K1 - K0) if K1 != K0 else 0
        beta = (T - T0) / (T1 - T0) if T1 != T0 else 0
        
        alpha = max(0, min(1, alpha))
        beta = max(0, min(1, beta))
        
        # Get corner values
        v00 = self.grid[i_T, i_K]
        v10 = self.grid[i_T, i_K + 1]
        v01 = self.grid[i_T + 1, i_K]
        v11 = self.grid[i_T + 1, i_K + 1]
        
        # Bilinear interpolation
        return ((1-alpha)*(1-beta)*v00 + alpha*(1-beta)*v10 +
                (1-alpha)*beta*v01 + alpha*beta*v11)
    
    def smile_at_maturity(self, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get volatility smile at specific maturity.
        
        Returns:
            (strikes, implied_vols) at nearest maturity
        """
        # Find nearest maturity
        i_T = np.argmin(np.abs(np.array(self.maturities) - T))
        
        vols = self.grid[i_T, :]
        valid = ~np.isnan(vols)
        
        return np.array(self.strikes)[valid], vols[valid]
    
    def term_structure(self, K: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get volatility term structure at specific strike.
        
        Returns:
            (maturities, implied_vols) at nearest strike
        """
        i_K = np.argmin(np.abs(np.array(self.strikes) - K))
        
        vols = self.grid[:, i_K]
        valid = ~np.isnan(vols)
        
        return np.array(self.maturities)[valid], vols[valid]
    
    def atm_vol(self, T: float, S: float) -> float:
        """Get ATM volatility at maturity T."""
        return self.interpolate(S, T)
    
    def compute_skew(self, T: float, S: float, dK: float = 0.05) -> float:
        """
        Compute volatility skew at maturity T.
        
        Skew Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Skew = [σ_IV(K_down) - σ_IV(K_up)] / (2·ΔK/S)
        
        Measures slope of smile in log-moneyness space.
        For equities, typically negative (downside skew).
        
        ═══════════════════════════════════════════════════════════════════════
        """
        K_down = S * (1 - dK)
        K_up = S * (1 + dK)
        
        vol_down = self.interpolate(K_down, T)
        vol_up = self.interpolate(K_up, T)
        
        return (vol_down - vol_up) / (2 * dK)
    
    def compute_convexity(self, T: float, S: float, dK: float = 0.05) -> float:
        """
        Compute smile convexity (butterfly) at maturity T.
        
        Convexity Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Convexity = [σ(K_down) + σ(K_up) - 2·σ(ATM)] / (ΔK/S)²
        
        Measures curvature of smile.
        Higher convexity = more pronounced smile.
        
        ═══════════════════════════════════════════════════════════════════════
        """
        K_down = S * (1 - dK)
        K_up = S * (1 + dK)
        K_atm = S
        
        vol_down = self.interpolate(K_down, T)
        vol_up = self.interpolate(K_up, T)
        vol_atm = self.interpolate(K_atm, T)
        
        return (vol_down + vol_up - 2*vol_atm) / (dK**2)


def create_sample_vol_surface(S0: float = 100) -> VolatilitySurface:
    """
    Create sample volatility surface for testing.
    
    Generates a realistic equity volatility surface with:
    - Negative skew (higher IV for lower strikes)
    - Smile curvature
    - Term structure (declining with maturity)
    """
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    maturities = np.array([0.083, 0.25, 0.5, 1.0, 2.0])  # 1M, 3M, 6M, 1Y, 2Y
    
    quotes = []
    
    for T in maturities:
        # ATM vol with term structure
        atm_vol = 0.20 + 0.05 * np.exp(-T)  # Higher for short maturities
        
        for K in strikes:
            moneyness = np.log(K / S0)
            
            # Add skew and smile
            skew = -0.10 * moneyness / np.sqrt(T + 0.1)  # Negative skew
            smile = 0.02 * moneyness**2 / (T + 0.1)  # Curvature
            
            iv = atm_vol + skew + smile
            iv = max(0.05, min(iv, 0.80))  # Bounds
            
            quotes.append(VolQuote(
                strike=K,
                maturity=T,
                implied_vol=iv
            ))
    
    return VolatilitySurface(quotes)
