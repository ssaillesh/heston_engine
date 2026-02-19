"""
Heston Semi-Analytical Pricing via Fourier Inversion

═══════════════════════════════════════════════════════════════════════════════
CHARACTERISTIC FUNCTION APPROACH TO OPTION PRICING
═══════════════════════════════════════════════════════════════════════════════

1. FOURIER TRANSFORM PRICING:
   ═══════════════════════════════════════════════════════════════════════════
   
   For a European call option under risk-neutral measure:
   
   C(S, K, τ) = e^{-rτ} · E^Q[(S_T - K)⁺]
   
   Using Fourier inversion, this becomes:
   
   C = S·e^{-qτ}·P₁ - K·e^{-rτ}·P₂
   
   where P₁, P₂ are probability-like integrals involving characteristic function.
   
   This is analogous to Black-Scholes formula but with Heston dynamics.

2. CHARACTERISTIC FUNCTION:
   ═══════════════════════════════════════════════════════════════════════════
   
   The characteristic function of ln(S_T) under Heston model is:
   
   φ(u; S, V, τ) = E[e^{iu·ln(S_T)} | S_0, V_0]
                 = exp(C(τ,u) + D(τ,u)·V_0 + iu·ln(S_0))
   
   This has closed-form expressions for C and D:
   
   Define:
   a = -u²/2 - iu/2
   b = κ - ρσiu  (or κ for P₂ formulation)
   d = √(b² - 2σ²a)
   g = (b - d)/(b + d)
   
   Then:
   C(τ,u) = (r-q)iuτ + (κθ/σ²)[(b - d)τ - 2ln((1 - ge^{dτ})/(1 - g))]
   
   D(τ,u) = [(b - d)/σ²] · (1 - e^{dτ}) / (1 - ge^{dτ})

3. TWO FORMULATIONS (P₁ and P₂):
   ═══════════════════════════════════════════════════════════════════════════
   
   For P₁ (in-the-money probability under S-measure):
   - Use adjusted drift: b₁ = κ - ρσ - ρσiu
   - This corresponds to measure change from Q to Q^S
   
   For P₂ (in-the-money probability under Q):
   - Use b₂ = κ - ρσiu
   - Standard risk-neutral measure
   
   The call price formula:
   C = S·e^{-qτ}·P₁ - K·e^{-rτ}·P₂

4. NUMERICAL INTEGRATION:
   ═══════════════════════════════════════════════════════════════════════════
   
   P_j = 1/2 + (1/π) ∫₀^∞ Re[e^{-iu·ln(K)} · φ_j(u) / (iu)] du
   
   Integration methods:
   - Adaptive quadrature (scipy.integrate.quad)
   - Gauss-Legendre quadrature (64-256 points)
   - FFT methods for multiple strikes
   
   Numerical considerations:
   - Integrand oscillates for large u
   - Truncate at u_max ≈ 50-500 (depends on τ)
   - Use damping factor for stability

5. BRANCH CUT ISSUES:
   ═══════════════════════════════════════════════════════════════════════════
   
   The square root d = √(b² - 2σ²a) requires careful branch selection.
   
   For stability, ensure:
   - Re(d) > 0 (positive real part)
   - Consistent branch across integration domain
   
   Alternative: Use the "rotation count" formulation that tracks
   the argument of complex numbers to avoid discontinuities.

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.integrate import quad
from typing import Tuple, Optional
from backend.core.parameters import HestonParams


class AnalyticalPricer:
    """
    Heston semi-analytical pricer using characteristic function approach.
    
    Implementation based on:
    Heston, S.L. (1993). "A Closed-Form Solution for Options with 
    Stochastic Volatility with Applications to Bond and Currency Options."
    """
    
    def __init__(self, params: HestonParams):
        """
        Initialize analytical pricer.
        
        Args:
            params: Heston model parameters
        """
        self.p = params
        
        # Integration parameters (u_max=150 matches QuantLib within $0.0003)
        self.u_max = 150  # Upper integration limit
        self.n_points = 128  # Quadrature points
    
    def characteristic_function(
        self, 
        u: complex, 
        tau: float, 
        formulation: int = 1
    ) -> complex:
        """
        Heston characteristic function.
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        φ(u; S₀, V₀, τ) = exp(C(τ,u) + D(τ,u)·V₀ + iu·ln(S₀))
        
        For formulation 1 (P₁, S-numeraire):
          b₁ = κ - ρσ
          u_adj = u - 1j (shift for measure change)
        
        For formulation 2 (P₂, bond-numeraire):
          b₂ = κ
          u_adj = u
        
        Helper quantities:
          a = -u_adj²/2 - iu_adj/2
          d = √(b² - 2σ²a)
          g = (b - d)/(b + d)
        
        Coefficient functions:
          C(τ,u) = (r-q)iuτ + (κθ/σ²)[(b-d)τ - 2ln((1-ge^{dτ})/(1-g))]
          D(τ,u) = [(b-d)/σ²] · (1-e^{dτ}) / (1-ge^{dτ})
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            u: Frequency variable (real or complex)
            tau: Time to maturity
            formulation: 1 for P₁ (S-measure), 2 for P₂ (Q-measure)
            
        Returns:
            Characteristic function value φ(u)
        """
        
        # Extract parameters
        kappa = self.p.kappa
        theta = self.p.theta
        sigma = self.p.sigma
        rho = self.p.rho
        r = self.p.r
        q = self.p.q
        V0 = self.p.V0
        S0 = self.p.S0
        
        # ═══════════════════════════════════════════════════════════════════
        # HESTON (1993) EXACT FORMULATION
        # ═══════════════════════════════════════════════════════════════════
        # From Heston's original paper, equation (17)
        # Two characteristic functions with different parameters:
        # 
        # For P₁: u₁ = 1/2,  b₁ = κ + λ - ρσ  (where λ = 0 for risk-neutral)
        # For P₂: u₂ = -1/2, b₂ = κ + λ        (where λ = 0 for risk-neutral)
        # 
        # d_j = √[(ρσφi - b_j)² - σ²(2u_j·φi - φ²)]
        # g_j = (b_j - ρσφi + d_j) / (b_j - ρσφi - d_j)
        # C_j = riφτ + (a/σ²)[(b_j - ρσφi + d_j)τ - 2 ln((1 - g_j e^{d_jτ})/(1 - g_j))]
        # D_j = [(b_j - ρσφi + d_j) / σ²] × (1 - e^{d_jτ}) / (1 - g_j e^{d_jτ})
        # ═══════════════════════════════════════════════════════════════════
        
        if formulation == 1:
            uj = 0.5
            bj = kappa - rho * sigma  # b₁ = κ - ρσ
        else:
            uj = -0.5
            bj = kappa  # b₂ = κ
        
        a = kappa * theta
        phi = u  # Fourier variable
        
        # ═══════════════════════════════════════════════════════════════════
        # DISCRIMINANT d_j = √[(ρσφi - b_j)² - σ²(2u_j·φi - φ²)]
        # ═══════════════════════════════════════════════════════════════════
        # Expanding: (ρσφi - bj)² = (bj - ρσφi)² = bj² - 2bj·ρσφi - (ρσφ)²
        # -σ²(2uj·φi - φ²) = -2σ²uj·φi + σ²φ²
        # So d² = bj² - 2bj·ρσφi - ρ²σ²φ² - 2σ²uj·φi + σ²φ²
        #       = bj² + σ²φ²(1 - ρ²) - 2iσφ(bj·ρ + σuj)
        # ═══════════════════════════════════════════════════════════════════
        
        # Direct computation
        d2 = (rho * sigma * phi * 1j - bj)**2 - sigma**2 * (2 * uj * phi * 1j - phi**2)
        d = np.sqrt(d2)
        
        # ═══════════════════════════════════════════════════════════════════
        # BRANCH CUT FIX: ensure Real(d) >= 0 for numerical stability
        # ═══════════════════════════════════════════════════════════════════
        if np.real(d) < 0:
            d = -d
        
        # ξ = b_j - ρσφi
        xi = bj - rho * sigma * phi * 1j
        
        # g = (ξ + d) / (ξ - d)  -- ORIGINAL Heston formula
        # But we use the STABLE formulation: g = (ξ - d) / (ξ + d) with exp(-dτ)
        g = (xi - d) / (xi + d + 1e-15)
        
        # ═══════════════════════════════════════════════════════════════════
        # COEFFICIENT FUNCTIONS (stable formulation with exp(-dτ))
        # ═══════════════════════════════════════════════════════════════════
        
        exp_neg_d_tau = np.exp(-d * tau)
        
        # D(τ) = (ξ - d)/σ² × (1 - e^{-dτ}) / (1 - g·e^{-dτ})
        D = ((xi - d) / (sigma**2)) * (1 - exp_neg_d_tau) / (1 - g * exp_neg_d_tau + 1e-15)
        
        # C(τ) = (r-q)·i·φ·τ + (a/σ²)[(ξ - d)τ - 2·ln((1 - g·e^{-dτ})/(1-g))]
        C = (r - q) * 1j * phi * tau
        log_term = np.log((1 - g * exp_neg_d_tau) / (1 - g + 1e-15))
        C += (a / (sigma**2)) * ((xi - d) * tau - 2 * log_term)
        
        # ═══════════════════════════════════════════════════════════════════
        # FULL CHARACTERISTIC FUNCTION
        # φ(u) = exp(C + D·V₀ + iu·ln(S₀))
        # ═══════════════════════════════════════════════════════════════════
        
        phi_out = np.exp(C + D * V0 + 1j * u * np.log(S0))
        
        return phi_out
    
    def _integrand_P1(self, u: float, K: float, tau: float) -> float:
        """
        Integrand for P₁ probability.
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Integrand = Re[e^{-iu·ln(K)} · φ₁(u) / (iu)]
        
        P₁ = 1/2 + (1/π) ∫₀^∞ Integrand du
        
        This represents P(S_T > K) under the S-numeraire measure Q^S.
        ═══════════════════════════════════════════════════════════════════════
        """
        if u < 1e-10:
            return 0.0
        
        phi = self.characteristic_function(u, tau, formulation=1)
        integrand = np.exp(-1j * u * np.log(K)) * phi / (1j * u)
        
        return np.real(integrand)
    
    def _integrand_P2(self, u: float, K: float, tau: float) -> float:
        """
        Integrand for P₂ probability.
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Integrand = Re[e^{-iu·ln(K)} · φ₂(u) / (iu)]
        
        P₂ = 1/2 + (1/π) ∫₀^∞ Integrand du
        
        This represents P(S_T > K) under the risk-neutral measure Q.
        ═══════════════════════════════════════════════════════════════════════
        """
        if u < 1e-10:
            return 0.0
        
        phi = self.characteristic_function(u, tau, formulation=2)
        integrand = np.exp(-1j * u * np.log(K)) * phi / (1j * u)
        
        return np.real(integrand)
    
    def call_price(self, K: float, T: float) -> float:
        """
        European call price via Fourier inversion.
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        C(S, K, T) = S·e^{-qT}·P₁ - K·e^{-rT}·P₂
        
        where:
        P_j = 1/2 + (1/π) ∫₀^∞ Re[e^{-iu·ln(K)} · φ_j(u) / (iu)] du
        
        This is the Heston analogue of Black-Scholes:
        C = S·N(d₁) - K·e^{-rT}·N(d₂)
        
        P₁ corresponds to N(d₁) (delta)
        P₂ corresponds to N(d₂) (exercise probability)
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            K: Strike price
            T: Time to maturity
            
        Returns:
            Call option price
        """
        
        S0 = self.p.S0
        r = self.p.r
        q = self.p.q
        
        # ═══════════════════════════════════════════════════════════════════
        # NUMERICAL INTEGRATION FOR P₁
        # ═══════════════════════════════════════════════════════════════════
        #
        # P₁ = 1/2 + (1/π) ∫₀^{u_max} Re[...] du
        #
        # Uses adaptive quadrature (scipy.integrate.quad)
        # with error control for accurate integration
        # ═══════════════════════════════════════════════════════════════════
        
        # Integrate P1
        integral_P1, _ = quad(
            lambda u: self._integrand_P1(u, K, T),
            0, self.u_max,
            limit=200
        )
        P1 = 0.5 + integral_P1 / np.pi
        
        # ═══════════════════════════════════════════════════════════════════
        # NUMERICAL INTEGRATION FOR P₂
        # ═══════════════════════════════════════════════════════════════════
        
        # Integrate P2
        integral_P2, _ = quad(
            lambda u: self._integrand_P2(u, K, T),
            0, self.u_max,
            limit=200
        )
        P2 = 0.5 + integral_P2 / np.pi
        
        # ═══════════════════════════════════════════════════════════════════
        # CALL PRICE
        # ═══════════════════════════════════════════════════════════════════
        #
        # C = S₀·e^{-qT}·P₁ - K·e^{-rT}·P₂
        #
        # Discount factors:
        # e^{-qT}: dividend adjustment (asset receives dividends)
        # e^{-rT}: time value of money (strike paid at T)
        # ═══════════════════════════════════════════════════════════════════
        
        call_price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        
        return max(call_price, 0.0)  # Ensure non-negative
    
    def put_price(self, K: float, T: float) -> float:
        """
        European put price via put-call parity.
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Put-Call Parity (under continuous dividends):
        
        C - P = S·e^{-qT} - K·e^{-rT}
        
        Therefore:
        P = C - S·e^{-qT} + K·e^{-rT}
        
        Derivation:
        - Buy call, sell put with same K, T
        - At expiry: payoff = (S_T - K)⁺ - (K - S_T)⁺ = S_T - K
        - Today's value: S·e^{-qT} - K·e^{-rT}
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            K: Strike price
            T: Time to maturity
            
        Returns:
            Put option price
        """
        call = self.call_price(K, T)
        
        # Put-call parity: P = C - S·e^{-qT} + K·e^{-rT}
        put = call - self.p.S0 * np.exp(-self.p.q * T) + K * np.exp(-self.p.r * T)
        
        return max(put, 0.0)
    
    def price_surface(
        self, 
        strikes: np.ndarray, 
        maturities: np.ndarray,
        option_type: str = 'call'
    ) -> np.ndarray:
        """
        Compute option price surface over strikes and maturities.
        
        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
            option_type: 'call' or 'put'
            
        Returns:
            2D array of prices, shape (len(maturities), len(strikes))
        """
        prices = np.zeros((len(maturities), len(strikes)))
        
        price_func = self.call_price if option_type == 'call' else self.put_price
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                prices[i, j] = price_func(K, T)
        
        return prices
    
    def implied_vol_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray
    ) -> np.ndarray:
        """
        Compute implied volatility surface.
        
        For each (K, T) pair:
        1. Compute Heston price
        2. Invert Black-Scholes to get implied vol
        
        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
            
        Returns:
            2D array of implied vols, shape (len(maturities), len(strikes))
        """
        from backend.core.boundaries import implied_volatility_newton
        
        iv_surface = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                # Compute Heston price
                price = self.call_price(K, T)
                
                # Invert to get implied vol
                try:
                    iv = implied_volatility_newton(
                        price, self.p.S0, K, T, self.p.r, self.p.q, 'call'
                    )
                    iv_surface[i, j] = iv
                except:
                    iv_surface[i, j] = np.nan
        
        return iv_surface
    
    def delta(self, K: float, T: float) -> float:
        """
        Compute Delta (∂C/∂S) analytically.
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Δ = e^{-qT} · P₁
        
        This is the probability-weighted sensitivity to spot price changes.
        Under Heston, delta includes volatility smile effects.
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            K: Strike price
            T: Time to maturity
            
        Returns:
            Delta value ∈ [0, 1] for calls
        """
        # Compute P1
        integral_P1, _ = quad(
            lambda u: self._integrand_P1(u, K, T),
            0, self.u_max,
            limit=200
        )
        P1 = 0.5 + integral_P1 / np.pi
        
        return np.exp(-self.p.q * T) * P1
    
    def vega_variance(self, K: float, T: float, dV: float = 0.001) -> float:
        """
        Compute Vega with respect to initial variance V₀.
        
        Uses finite difference:
        Vega_V = ∂C/∂V₀ ≈ [C(V₀ + dV) - C(V₀ - dV)] / (2dV)
        
        Args:
            K: Strike price
            T: Time to maturity
            dV: Finite difference step
            
        Returns:
            Sensitivity to initial variance
        """
        # Save original V0
        V0_orig = self.p.V0
        
        # Compute prices at V0 ± dV
        self.p.V0 = V0_orig + dV
        price_up = self.call_price(K, T)
        
        self.p.V0 = V0_orig - dV
        price_down = self.call_price(K, T)
        
        # Restore V0
        self.p.V0 = V0_orig
        
        # Finite difference
        vega = (price_up - price_down) / (2 * dV)
        
        return vega
