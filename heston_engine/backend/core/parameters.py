"""
Heston Model Parameters with Mathematical Validation

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION - HESTON STOCHASTIC VOLATILITY MODEL
═══════════════════════════════════════════════════════════════════════════════

The Heston model describes asset price dynamics under the risk-neutral measure:

1. ASSET PRICE SDE:
   dS_t = (r - q)S_t dt + √V_t S_t dW_S^t
   
   Where:
   - S_t: Asset price at time t
   - r: Risk-free interest rate (continuous compounding)
   - q: Continuous dividend yield
   - V_t: Instantaneous variance (volatility² at time t)
   - dW_S^t: Brownian motion driving price

2. VARIANCE SDE (CIR Process):
   dV_t = κ(θ - V_t)dt + σ√V_t dW_V^t
   
   Where:
   - κ (kappa) > 0: Mean reversion speed
     * Higher κ → faster reversion to long-term mean
     * Typical range: [0.5, 10]
   
   - θ (theta) > 0: Long-term variance level
     * V_t reverts to θ over time
     * Typical range: [0.01, 0.25] (i.e., 10%-50% volatility)
   
   - σ (sigma) > 0: Volatility of volatility (vol-of-vol)
     * Controls variance path roughness
     * Typical range: [0.1, 1.0]
   
   - dW_V^t: Brownian motion driving variance

3. CORRELATION STRUCTURE:
   E[dW_S^t · dW_V^t] = ρ dt
   
   Where ρ ∈ [-1, 1] is the correlation:
   - ρ < 0 (typical for equities): Negative price shocks → higher volatility
     This creates the "leverage effect" or volatility skew
   - ρ > 0 (some commodities): Price increases → higher volatility
   - Typical equity range: [-0.9, -0.3]

4. FELLER CONDITION (Variance Non-Negativity):
   2κθ > σ²
   
   Derivation:
   - For CIR process, variance can reach zero
   - If 2κθ > σ², the drift dominates diffusion near zero
   - Probability of V_t hitting zero is zero when condition holds
   
   Feller ratio: F = 2κθ/σ²
   - F > 1: Condition satisfied, variance stays positive
   - F ≤ 1: Variance can hit zero, may need reflection scheme

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import warnings


@dataclass
class HestonParams:
    """
    Container for Heston model parameters with validation.
    
    All parameters are stored and validated according to mathematical constraints.
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # Mean Reversion Parameters (Variance Process)
    # ═══════════════════════════════════════════════════════════════════════
    
    kappa: float  # κ: Speed of mean reversion in variance
                  # Units: 1/time
                  # Higher κ → V_t reverts faster to θ
    
    theta: float  # θ: Long-term variance level (equilibrium)
                  # Units: (price return)² per unit time
                  # Annualized volatility ≈ √θ
    
    sigma: float  # σ: Volatility of volatility (vol-of-vol)
                  # Units: √(variance/time)
                  # Controls roughness of variance paths
    
    rho: float    # ρ: Correlation between price and variance Brownians
                  # Dimensionless, range [-1, 1]
                  # ρ < 0 creates volatility skew (leverage effect)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Market Parameters
    # ═══════════════════════════════════════════════════════════════════════
    
    r: float      # Risk-free interest rate (continuous compounding)
                  # Units: 1/time (typically annualized)
    
    q: float      # Continuous dividend yield
                  # Units: 1/time (typically annualized)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Initial Conditions
    # ═══════════════════════════════════════════════════════════════════════
    
    S0: float     # Initial spot price S(0)
                  # Units: currency (e.g., USD)
    
    V0: float     # Initial variance V(0)
                  # Units: (price return)² per unit time
                  # Initial volatility = √V0
    
    def __post_init__(self):
        """
        Validate all parameters against mathematical constraints.
        
        Mathematical Requirements:
        ═══════════════════════════════════════════════════════════════════════
        
        1. κ > 0: Mean reversion speed must be positive
           - κ = 0 would mean no reversion (random walk variance)
           - κ < 0 would cause explosive variance
        
        2. θ > 0: Long-term variance must be positive
           - θ ≤ 0 is meaningless (negative variance)
        
        3. σ > 0: Vol-of-vol must be positive
           - σ = 0 would make variance deterministic
           - σ < 0 is meaningless
        
        4. -1 ≤ ρ ≤ 1: Valid correlation coefficient
           - ρ = ±1 gives perfectly correlated Brownians
           - |ρ| > 1 is mathematically invalid
        
        5. S₀ > 0, V₀ > 0: Positive initial values
           - S₀ ≤ 0 is meaningless (negative price)
           - V₀ ≤ 0 would cause imaginary volatility
        
        6. FELLER CONDITION: 2κθ > σ²
           - Ensures variance process never reaches zero
           - If violated, need truncation/reflection scheme
        ═══════════════════════════════════════════════════════════════════════
        """
        
        # Parameter positivity checks
        assert self.kappa > 0, f"κ must be positive, got {self.kappa}"
        assert self.theta > 0, f"θ must be positive, got {self.theta}"
        assert self.sigma > 0, f"σ must be positive, got {self.sigma}"
        
        # Correlation bound check
        assert -1 <= self.rho <= 1, f"ρ must be in [-1, 1], got {self.rho}"
        
        # Initial condition checks
        assert self.S0 > 0, f"S₀ must be positive, got {self.S0}"
        assert self.V0 > 0, f"V₀ must be positive, got {self.V0}"
        
        # ═══════════════════════════════════════════════════════════════════
        # FELLER CONDITION CHECK
        # ═══════════════════════════════════════════════════════════════════
        # 
        # The Feller condition ensures the variance process V_t > 0 a.s.
        # 
        # For CIR process: dV = κ(θ - V)dt + σ√V dW
        # 
        # Near V = 0:
        #   - Drift term: κθ (pulling away from zero)
        #   - Diffusion term: ~ σ√V → 0 as V → 0
        # 
        # The condition 2κθ > σ² ensures drift dominates diffusion
        # at the boundary, preventing V from reaching zero.
        # 
        # Mathematically: P(inf{t: V_t = 0} < ∞) = 0 when 2κθ > σ²
        # ═══════════════════════════════════════════════════════════════════
        
        feller_lhs = 2 * self.kappa * self.theta  # = 2κθ
        feller_rhs = self.sigma ** 2               # = σ²
        self.feller_ratio = feller_lhs / feller_rhs  # F = 2κθ/σ²
        self.feller_satisfied = self.feller_ratio > 1.0
        
        if not self.feller_satisfied:
            warnings.warn(
                f"⚠️  FELLER CONDITION VIOLATED!\n"
                f"    2κθ = {feller_lhs:.6f}\n"
                f"    σ²  = {feller_rhs:.6f}\n"
                f"    Ratio 2κθ/σ² = {self.feller_ratio:.4f} ≤ 1\n"
                f"    Variance V_t may reach zero with positive probability.\n"
                f"    Consider: increasing κ or θ, or decreasing σ."
            )
        
        # Compute derived quantities
        self._compute_derived()
    
    def _compute_derived(self):
        """
        Compute derived quantities useful for analysis.
        
        Derived Quantities:
        ═══════════════════════════════════════════════════════════════════════
        
        1. Long-term volatility: σ_∞ = √θ
           - Annualized volatility in equilibrium
        
        2. Initial volatility: σ₀ = √V₀
           - Current instantaneous volatility
        
        3. Variance half-life: τ_{1/2} = ln(2)/κ
           - Time for variance to move halfway to θ
        
        4. Variance mean and variance under stationary distribution:
           - E[V_∞] = θ
           - Var[V_∞] = θσ²/(2κ)
        
        5. Forward variance curve (expectation):
           - E[V_t | V_0] = θ + (V₀ - θ)e^{-κt}
        ═══════════════════════════════════════════════════════════════════════
        """
        
        # Volatility levels
        self.long_term_vol = np.sqrt(self.theta)   # σ_∞ = √θ
        self.initial_vol = np.sqrt(self.V0)        # σ₀ = √V₀
        
        # Mean reversion characteristics
        # Half-life: time for E[V_t - θ] to decay to half
        # E[V_t - θ] = (V₀ - θ)e^{-κt}
        # Setting e^{-κt} = 0.5 → t = ln(2)/κ
        self.variance_halflife = np.log(2) / self.kappa
        
        # Stationary variance (long-run distribution of V)
        # V_∞ ~ Gamma with mean θ and variance θσ²/(2κ)
        self.stationary_variance_mean = self.theta
        self.stationary_variance_var = self.theta * self.sigma**2 / (2 * self.kappa)
    
    def expected_variance(self, t: float) -> float:
        """
        Expected variance at time t given V₀.
        
        Formula:
        ═══════════════════════════════════════════════════════════════════════
        E[V_t | V₀] = θ + (V₀ - θ)e^{-κt}
        
        Derivation:
        - Solve E[dV_t] = κ(θ - E[V_t])dt (drift-only ODE)
        - d(E[V_t] - θ)/dt = -κ(E[V_t] - θ)
        - Solution: E[V_t] - θ = (V₀ - θ)e^{-κt}
        
        Properties:
        - As t → ∞: E[V_t] → θ (convergence to long-term mean)
        - At t = 0: E[V₀] = V₀
        - Rate of convergence determined by κ
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            t: Time horizon
            
        Returns:
            Expected variance E[V_t]
        """
        return self.theta + (self.V0 - self.theta) * np.exp(-self.kappa * t)
    
    def variance_confidence_interval(self, t: float, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Confidence interval for variance at time t.
        
        Uses approximate normal distribution for large t.
        
        Args:
            t: Time horizon
            confidence: Confidence level (default 95%)
            
        Returns:
            (lower, upper) bounds
        """
        from scipy import stats
        
        mean_V = self.expected_variance(t)
        
        # Variance of V_t (approximate for moderate t)
        # For CIR process, exact formula involves confluent hypergeometric functions
        # Use stationary approximation for simplicity
        var_V = self.stationary_variance_var * (1 - np.exp(-2 * self.kappa * t))
        
        std_V = np.sqrt(max(var_V, 1e-10))
        z = stats.norm.ppf((1 + confidence) / 2)
        
        lower = max(0, mean_V - z * std_V)
        upper = mean_V + z * std_V
        
        return lower, upper
    
    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary for serialization."""
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'rho': self.rho,
            'r': self.r,
            'q': self.q,
            'S0': self.S0,
            'V0': self.V0
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'HestonParams':
        """Create HestonParams from dictionary."""
        return cls(
            kappa=d['kappa'],
            theta=d['theta'],
            sigma=d['sigma'],
            rho=d['rho'],
            r=d['r'],
            q=d['q'],
            S0=d['S0'],
            V0=d['V0']
        )
    
    def __repr__(self) -> str:
        feller_status = "✓" if self.feller_satisfied else "✗"
        return (
            f"HestonParams(\n"
            f"  κ={self.kappa:.4f}, θ={self.theta:.4f}, σ={self.sigma:.4f}, ρ={self.rho:.4f}\n"
            f"  r={self.r:.4f}, q={self.q:.4f}\n"
            f"  S₀={self.S0:.2f}, V₀={self.V0:.4f}\n"
            f"  Long-term vol: {self.long_term_vol*100:.1f}%\n"
            f"  Feller ratio: {self.feller_ratio:.3f} {feller_status}\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TYPICAL PARAMETER SETS FOR TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def get_default_params() -> HestonParams:
    """
    Default parameters for testing.
    
    These parameters satisfy the Feller condition and produce
    realistic equity-like volatility dynamics.
    """
    return HestonParams(
        kappa=2.0,    # Moderate mean reversion
        theta=0.04,   # 20% long-term volatility
        sigma=0.3,    # Moderate vol-of-vol
        rho=-0.7,     # Strong negative correlation (equity-like)
        r=0.05,       # 5% risk-free rate
        q=0.02,       # 2% dividend yield
        S0=100.0,     # Spot price $100
        V0=0.04       # Initial volatility = 20%
    )


def get_heston_1993_params() -> HestonParams:
    """
    Parameters from Heston's 1993 paper (Table 1, Example 1).
    
    Reference:
    Heston, S. L. (1993). "A Closed-Form Solution for Options with 
    Stochastic Volatility with Applications to Bond and Currency Options."
    The Review of Financial Studies, 6(2), 327-343.
    """
    return HestonParams(
        kappa=2.0,
        theta=0.01,
        sigma=0.1,
        rho=0.0,
        r=0.0,
        q=0.0,
        S0=100.0,
        V0=0.01
    )


def get_calibration_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Return typical bounds for calibration optimization.
    
    Calibration Constraints:
    ═══════════════════════════════════════════════════════════════════════════
    
    κ ∈ [0.01, 15]:
      - Lower: Very slow mean reversion
      - Upper: Very fast reversion (nearly deterministic)
    
    θ ∈ [0.001, 1.0]:
      - Lower: ~3% volatility floor
      - Upper: 100% volatility ceiling
    
    σ ∈ [0.01, 3.0]:
      - Lower: Nearly deterministic variance
      - Upper: Extreme vol-of-vol
    
    ρ ∈ [-0.99, -0.01]:
      - Typically negative for equities
      - Avoid ±1 for numerical stability
    
    V₀ ∈ [0.001, 1.0]:
      - Same range as θ
    ═══════════════════════════════════════════════════════════════════════════
    """
    return {
        'kappa': (0.01, 15.0),
        'theta': (0.001, 1.0),
        'sigma': (0.01, 3.0),
        'rho': (-0.99, -0.01),
        'V0': (0.001, 1.0)
    }
