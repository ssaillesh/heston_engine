"""
Greeks Calculator for Heston Model

═══════════════════════════════════════════════════════════════════════════════
OPTION GREEKS UNDER HESTON STOCHASTIC VOLATILITY
═══════════════════════════════════════════════════════════════════════════════

Under Heston model, Greeks depend on both S and V (variance state).
This adds complexity compared to Black-Scholes.

1. FIRST-ORDER GREEKS:
   ═══════════════════════════════════════════════════════════════════════════
   
   Delta (Δ) = ∂C/∂S
   - Sensitivity to spot price changes
   - Under Heston: Delta depends on current variance V
   - For calls: 0 < Δ < 1, increases with S
   
   Vega_V = ∂C/∂V₀
   - Sensitivity to initial variance
   - Note: Different from volatility vega in Black-Scholes
   
   Theta (Θ) = -∂C/∂τ = ∂C/∂t
   - Time decay
   - Typically negative for long options
   
   Rho (ρ) = ∂C/∂r
   - Sensitivity to interest rate

2. SECOND-ORDER GREEKS:
   ═══════════════════════════════════════════════════════════════════════════
   
   Gamma (Γ) = ∂²C/∂S²
   - Curvature of delta
   - Measures hedging cost from large moves
   
   Vanna = ∂²C/∂S∂V = ∂Δ/∂V
   - Cross-sensitivity to spot and variance
   - Important for volatility smile dynamics
   
   Volga = ∂²C/∂V²
   - Curvature in variance direction
   - Measures convexity of vega

3. COMPUTATION METHODS:
   ═══════════════════════════════════════════════════════════════════════════
   
   Finite difference:
   Δ ≈ [C(S+h) - C(S-h)] / (2h)
   Γ ≈ [C(S+h) - 2C(S) + C(S-h)] / h²
   
   Analytical (via characteristic function):
   Δ = e^{-qτ}·P₁  (call option)
   
   PDE-based:
   Extract from solution grid using finite differences

4. VOLATILITY VEGAS:
   ═══════════════════════════════════════════════════════════════════════════
   
   Under Heston, multiple "vega" measures:
   
   ∂C/∂V₀: Sensitivity to initial variance
   ∂C/∂θ: Sensitivity to long-term variance
   ∂C/∂κ: Sensitivity to mean reversion speed
   ∂C/∂σ: Sensitivity to vol-of-vol
   
   For hedging, the most relevant are ∂C/∂V₀ and ∂C/∂θ.

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, Tuple, Optional
from backend.core.parameters import HestonParams
from backend.solvers.analytical import AnalyticalPricer


class GreeksCalculator:
    """
    Greeks calculator for Heston model options.
    
    Computes first and second-order sensitivities using
    finite difference methods on the analytical pricer.
    """
    
    def __init__(self, params: HestonParams):
        """
        Initialize Greeks calculator.
        
        Args:
            params: Heston model parameters
        """
        self.params = params
        self.pricer = AnalyticalPricer(params)
        
        # Default bump sizes for finite differences
        self.dS = 0.01  # 1% spot bump
        self.dV = 0.001  # Variance bump
        self.dt = 0.001  # Time bump (in years)
        self.dr = 0.0001  # Rate bump
    
    def delta(self, K: float, T: float, option_type: str = 'call') -> float:
        """
        Compute Delta via finite difference.
        
        Delta Calculation:
        ═══════════════════════════════════════════════════════════════════════
        
        Δ = ∂C/∂S ≈ [C(S+h) - C(S-h)] / (2h)
        
        Central difference: Second-order accurate in h
        
        For calls under Heston:
        - Delta depends on variance state V₀
        - Higher V₀ → Delta closer to 0.5 (more uncertainty)
        - Lower V₀ → Delta more extreme (0 or 1)
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            
        Returns:
            Delta value
        """
        S0 = self.params.S0
        h = S0 * self.dS  # Absolute bump
        
        # Save original S0
        S0_orig = self.params.S0
        
        # C(S + h)
        self.params.S0 = S0_orig + h
        pricer_up = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_up = pricer_up.call_price(K, T)
        else:
            price_up = pricer_up.put_price(K, T)
        
        # C(S - h)
        self.params.S0 = S0_orig - h
        pricer_down = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_down = pricer_down.call_price(K, T)
        else:
            price_down = pricer_down.put_price(K, T)
        
        # Restore S0
        self.params.S0 = S0_orig
        
        # Central difference
        delta = (price_up - price_down) / (2 * h)
        
        return delta
    
    def gamma(self, K: float, T: float, option_type: str = 'call') -> float:
        """
        Compute Gamma via finite difference.
        
        Gamma Calculation:
        ═══════════════════════════════════════════════════════════════════════
        
        Γ = ∂²C/∂S² ≈ [C(S+h) - 2C(S) + C(S-h)] / h²
        
        Gamma measures:
        - Curvature of delta (how delta changes with S)
        - Hedging cost from gamma trading
        - Maximum near ATM, decays for deep ITM/OTM
        
        ═══════════════════════════════════════════════════════════════════════
        """
        S0 = self.params.S0
        h = S0 * self.dS
        
        S0_orig = self.params.S0
        
        # C(S)
        if option_type == 'call':
            price_mid = self.pricer.call_price(K, T)
        else:
            price_mid = self.pricer.put_price(K, T)
        
        # C(S + h)
        self.params.S0 = S0_orig + h
        pricer_up = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_up = pricer_up.call_price(K, T)
        else:
            price_up = pricer_up.put_price(K, T)
        
        # C(S - h)
        self.params.S0 = S0_orig - h
        pricer_down = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_down = pricer_down.call_price(K, T)
        else:
            price_down = pricer_down.put_price(K, T)
        
        # Restore
        self.params.S0 = S0_orig
        self.pricer = AnalyticalPricer(self.params)
        
        # Second-order central difference
        gamma = (price_up - 2*price_mid + price_down) / (h**2)
        
        return gamma
    
    def vega(self, K: float, T: float, option_type: str = 'call') -> float:
        """
        Compute Vega (sensitivity to initial variance V₀).
        
        Vega Calculation:
        ═══════════════════════════════════════════════════════════════════════
        
        Vega_V = ∂C/∂V₀ ≈ [C(V₀+dV) - C(V₀-dV)] / (2dV)
        
        Note: This is ∂C/∂V₀, not ∂C/∂σ (vol-of-vol)
        
        To convert to volatility vega:
        ∂C/∂σ_implied ≈ 2√V₀ · ∂C/∂V₀
        
        ═══════════════════════════════════════════════════════════════════════
        """
        V0_orig = self.params.V0
        dV = self.dV
        
        # C(V₀ + dV)
        self.params.V0 = V0_orig + dV
        pricer_up = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_up = pricer_up.call_price(K, T)
        else:
            price_up = pricer_up.put_price(K, T)
        
        # C(V₀ - dV)
        self.params.V0 = max(V0_orig - dV, 1e-6)
        pricer_down = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_down = pricer_down.call_price(K, T)
        else:
            price_down = pricer_down.put_price(K, T)
        
        # Restore
        self.params.V0 = V0_orig
        self.pricer = AnalyticalPricer(self.params)
        
        vega = (price_up - price_down) / (2 * dV)
        
        return vega
    
    def theta(self, K: float, T: float, option_type: str = 'call') -> float:
        """
        Compute Theta (time decay).
        
        Theta Calculation:
        ═══════════════════════════════════════════════════════════════════════
        
        Θ = -∂C/∂τ = ∂C/∂t ≈ -[C(τ+dτ) - C(τ-dτ)] / (2dτ)
        
        Theta is typically negative for long options
        (option loses value as time passes).
        
        Convention: Usually quoted per day (divide by 365)
        
        ═══════════════════════════════════════════════════════════════════════
        """
        dT = self.dt
        
        price_func = self.pricer.call_price if option_type == 'call' else self.pricer.put_price
        
        # C(τ + dτ)
        price_up = price_func(K, T + dT)
        
        # C(τ - dτ) - be careful if T - dT < 0
        if T - dT > 0.001:
            price_down = price_func(K, T - dT)
            # Central difference
            theta = -(price_up - price_down) / (2 * dT)
        else:
            # Forward difference near expiry
            price_mid = price_func(K, T)
            theta = -(price_up - price_mid) / dT
        
        return theta
    
    def rho(self, K: float, T: float, option_type: str = 'call') -> float:
        """
        Compute Rho (sensitivity to interest rate).
        
        Rho Calculation:
        ═══════════════════════════════════════════════════════════════════════
        
        ρ = ∂C/∂r ≈ [C(r+dr) - C(r-dr)] / (2dr)
        
        For calls: Rho > 0 (higher rates increase call value)
        For puts: Rho < 0 (higher rates decrease put value)
        
        Scales with T (longer maturity → larger rho)
        
        ═══════════════════════════════════════════════════════════════════════
        """
        r_orig = self.params.r
        dr = self.dr
        
        # C(r + dr)
        self.params.r = r_orig + dr
        pricer_up = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_up = pricer_up.call_price(K, T)
        else:
            price_up = pricer_up.put_price(K, T)
        
        # C(r - dr)
        self.params.r = r_orig - dr
        pricer_down = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_down = pricer_down.call_price(K, T)
        else:
            price_down = pricer_down.put_price(K, T)
        
        # Restore
        self.params.r = r_orig
        self.pricer = AnalyticalPricer(self.params)
        
        rho = (price_up - price_down) / (2 * dr)
        
        return rho
    
    def vanna(self, K: float, T: float, option_type: str = 'call') -> float:
        """
        Compute Vanna (cross-derivative ∂²C/∂S∂V).
        
        Vanna Calculation:
        ═══════════════════════════════════════════════════════════════════════
        
        Vanna = ∂²C/∂S∂V = ∂Δ/∂V
        
        Interpretation:
        - How delta changes with variance
        - Important for hedging smile dynamics
        
        Formula (mixed partial via finite difference):
        Vanna ≈ [C(S+h,V+dV) - C(S+h,V-dV) - C(S-h,V+dV) + C(S-h,V-dV)] / (4h·dV)
        
        ═══════════════════════════════════════════════════════════════════════
        """
        S0_orig = self.params.S0
        V0_orig = self.params.V0
        h = S0_orig * self.dS
        dV = self.dV
        
        # C(S+h, V+dV)
        self.params.S0 = S0_orig + h
        self.params.V0 = V0_orig + dV
        pricer_pp = AnalyticalPricer(self.params)
        if option_type == 'call':
            p_pp = pricer_pp.call_price(K, T)
        else:
            p_pp = pricer_pp.put_price(K, T)
        
        # C(S+h, V-dV)
        self.params.V0 = max(V0_orig - dV, 1e-6)
        pricer_pm = AnalyticalPricer(self.params)
        if option_type == 'call':
            p_pm = pricer_pm.call_price(K, T)
        else:
            p_pm = pricer_pm.put_price(K, T)
        
        # C(S-h, V+dV)
        self.params.S0 = S0_orig - h
        self.params.V0 = V0_orig + dV
        pricer_mp = AnalyticalPricer(self.params)
        if option_type == 'call':
            p_mp = pricer_mp.call_price(K, T)
        else:
            p_mp = pricer_mp.put_price(K, T)
        
        # C(S-h, V-dV)
        self.params.V0 = max(V0_orig - dV, 1e-6)
        pricer_mm = AnalyticalPricer(self.params)
        if option_type == 'call':
            p_mm = pricer_mm.call_price(K, T)
        else:
            p_mm = pricer_mm.put_price(K, T)
        
        # Restore
        self.params.S0 = S0_orig
        self.params.V0 = V0_orig
        self.pricer = AnalyticalPricer(self.params)
        
        vanna = (p_pp - p_pm - p_mp + p_mm) / (4 * h * dV)
        
        return vanna
    
    def volga(self, K: float, T: float, option_type: str = 'call') -> float:
        """
        Compute Volga (second derivative ∂²C/∂V²).
        
        Volga Calculation:
        ═══════════════════════════════════════════════════════════════════════
        
        Volga = ∂²C/∂V₀² ≈ [C(V+dV) - 2C(V) + C(V-dV)] / dV²
        
        Interpretation:
        - Curvature of option value in variance
        - Measures convexity of vega
        
        ═══════════════════════════════════════════════════════════════════════
        """
        V0_orig = self.params.V0
        dV = self.dV
        
        # C(V)
        if option_type == 'call':
            price_mid = self.pricer.call_price(K, T)
        else:
            price_mid = self.pricer.put_price(K, T)
        
        # C(V + dV)
        self.params.V0 = V0_orig + dV
        pricer_up = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_up = pricer_up.call_price(K, T)
        else:
            price_up = pricer_up.put_price(K, T)
        
        # C(V - dV)
        self.params.V0 = max(V0_orig - dV, 1e-6)
        pricer_down = AnalyticalPricer(self.params)
        if option_type == 'call':
            price_down = pricer_down.call_price(K, T)
        else:
            price_down = pricer_down.put_price(K, T)
        
        # Restore
        self.params.V0 = V0_orig
        self.pricer = AnalyticalPricer(self.params)
        
        volga = (price_up - 2*price_mid + price_down) / (dV**2)
        
        return volga
        
        return volga
    
    def all_greeks(self, K: float, T: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Compute all Greeks for an option.
        
        Returns:
            Dictionary with all Greek values
        """
        price_func = self.pricer.call_price if option_type == 'call' else self.pricer.put_price
        
        return {
            'price': price_func(K, T),
            'delta': self.delta(K, T, option_type),
            'gamma': self.gamma(K, T, option_type),
            'vega': self.vega(K, T, option_type),
            'theta': self.theta(K, T, option_type),
            'rho': self.rho(K, T, option_type),
            'vanna': self.vanna(K, T, option_type),
            'volga': self.volga(K, T, option_type)
        }
    
    def greek_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        greek: str = 'delta',
        option_type: str = 'call'
    ) -> np.ndarray:
        """
        Compute Greek surface over strikes and maturities.
        
        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
            greek: 'delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'volga'
            option_type: 'call' or 'put'
            
        Returns:
            2D array of Greek values, shape (len(maturities), len(strikes))
        """
        greek_func = getattr(self, greek)
        surface = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                surface[i, j] = greek_func(K, T, option_type)
        
        return surface
    
    def parameter_sensitivities(
        self,
        K: float,
        T: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Compute sensitivities to all Heston parameters.
        
        Returns sensitivities to:
        - κ (kappa): mean reversion speed
        - θ (theta): long-term variance
        - σ (sigma): vol-of-vol
        - ρ (rho): correlation
        
        These are useful for understanding model risk.
        """
        params_orig = {
            'kappa': self.params.kappa,
            'theta': self.params.theta,
            'sigma': self.params.sigma,
            'rho': self.params.rho
        }
        
        bumps = {
            'kappa': 0.1,
            'theta': 0.001,
            'sigma': 0.01,
            'rho': 0.01
        }
        
        price_func = self.pricer.call_price if option_type == 'call' else self.pricer.put_price
        sensitivities = {}
        
        for param_name, bump in bumps.items():
            # Bump up
            setattr(self.params, param_name, params_orig[param_name] + bump)
            self.pricer = AnalyticalPricer(self.params)
            price_up = price_func(K, T)
            
            # Bump down
            setattr(self.params, param_name, params_orig[param_name] - bump)
            self.pricer = AnalyticalPricer(self.params)
            price_down = price_func(K, T)
            
            # Restore
            setattr(self.params, param_name, params_orig[param_name])
            
            sensitivities[f'd_d{param_name}'] = (price_up - price_down) / (2 * bump)
        
        self.pricer = AnalyticalPricer(self.params)
        return sensitivities
