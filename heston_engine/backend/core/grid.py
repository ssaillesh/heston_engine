"""
Grid Generation for PDE Solver

═══════════════════════════════════════════════════════════════════════════════
SPATIAL AND TEMPORAL DISCRETIZATION FOR HESTON PDE
═══════════════════════════════════════════════════════════════════════════════

The Heston PDE is defined on a 2D spatial domain (S, V) × time domain [0, T].
For numerical solution, we discretize this domain into a grid.

1. LOG-PRICE TRANSFORMATION:
   ═══════════════════════════════════════════════════════════════════════════
   
   Instead of S, we use x = ln(S) for better numerical properties:
   
   Benefits:
   - Removes S-dependence from diffusion coefficient
   - Better resolution near ATM
   - Natural for percentage moves
   
   Inverse: S = e^x
   
   Derivative transformations:
   ∂/∂S = (1/S)∂/∂x
   ∂²/∂S² = (1/S²)(∂²/∂x² - ∂/∂x)

2. S-GRID (or x-grid) DESIGN:
   ═══════════════════════════════════════════════════════════════════════════
   
   Uniform in log-space:
   x_i = x_min + i·Δx,  i = 0, 1, ..., N_S - 1
   
   where:
   - x_min = ln(S_min), typically ln(0.2·S₀)
   - x_max = ln(S_max), typically ln(5·S₀)
   - Δx = (x_max - x_min)/(N_S - 1)
   
   This is equivalent to geometric spacing in S:
   S_i = S_min · exp(i·Δx)
   
   Typical values:
   - N_S = 100-200 points
   - S_min = 0.2·S₀ (far OTM put region)
   - S_max = 5·S₀ (far OTM call region)

3. V-GRID DESIGN:
   ═══════════════════════════════════════════════════════════════════════════
   
   Uniform spacing (can use non-uniform for better resolution near 0):
   V_j = V_min + j·ΔV,  j = 0, 1, ..., N_V - 1
   
   where:
   - V_min ≈ 10⁻⁸ (small positive, avoid exactly 0)
   - V_max ≈ 5θ (several times long-term variance)
   - ΔV = (V_max - V_min)/(N_V - 1)
   
   Typical values:
   - N_V = 50-100 points
   - V_min = 10⁻⁸ (numerically "zero")
   - V_max = 5·θ or max(1, 5θ)

4. TIME GRID:
   ═══════════════════════════════════════════════════════════════════════════
   
   Backward stepping from T to 0 (for European options):
   
   τ_n = n·Δτ,  n = 0, 1, ..., N_t
   
   where:
   - τ = T - t is time-to-maturity
   - τ = 0 at expiry, τ = T at valuation date
   - Δτ = T/N_t
   
   The PDE is solved backward: start at τ=0 (terminal condition)
   and step to τ=T (today's value).
   
   Typical values:
   - N_t = 100-500 time steps
   - Δτ should satisfy stability conditions

5. GRID STABILITY (CFL-type conditions):
   ═══════════════════════════════════════════════════════════════════════════
   
   For explicit schemes, need:
   Δτ ≤ C·min(Δx², ΔV²) / max(V·S², σ²·V)
   
   For Crank-Nicolson/ADI: unconditionally stable but may oscillate
   if Δτ too large relative to spatial steps.
   
   Rule of thumb for ADI:
   - Δτ ≈ Δx² / max_V for good accuracy
   - N_t ≈ T·(N_S)² / (x_max - x_min)² · V_max

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Tuple, Optional


class Grid:
    """
    2D spatial + time grid for Heston PDE solver.
    
    The grid discretizes:
    - S-dimension (price): using log-price x = ln(S)
    - V-dimension (variance): uniform spacing
    - τ-dimension (time-to-maturity): uniform backward stepping
    """
    
    def __init__(
        self,
        S_min: float,
        S_max: float,
        N_S: int,
        V_min: float,
        V_max: float,
        N_V: int,
        T: float,
        N_t: int
    ):
        """
        Create 2D spatial + time grid.
        
        Grid Construction:
        ═══════════════════════════════════════════════════════════════════════
        
        S-grid (log-price):
        - x_i = ln(S_min) + i·Δx for i = 0, ..., N_S-1
        - Δx = [ln(S_max) - ln(S_min)] / (N_S - 1)
        - S_i = exp(x_i)
        
        V-grid (uniform variance):
        - V_j = V_min + j·ΔV for j = 0, ..., N_V-1  
        - ΔV = (V_max - V_min) / (N_V - 1)
        
        τ-grid (time-to-maturity):
        - τ_n = n·Δτ for n = 0, ..., N_t
        - Δτ = T / N_t
        - τ = 0 corresponds to expiry (terminal condition)
        - τ = T corresponds to valuation date (t = 0)
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            S_min: Minimum price (typically 0.2*S0)
            S_max: Maximum price (typically 5*S0)
            N_S: Number of price grid points
            V_min: Minimum variance (small positive, ~1e-8)
            V_max: Maximum variance (typically 5*theta)
            N_V: Number of variance grid points
            T: Time to maturity
            N_t: Number of time steps
        """
        
        # Store parameters
        self.S_min, self.S_max, self.N_S = S_min, S_max, N_S
        self.V_min, self.V_max, self.N_V = V_min, V_max, N_V
        self.T, self.N_t = T, N_t
        
        # ═══════════════════════════════════════════════════════════════════
        # LOG-PRICE GRID (x = ln(S))
        # ═══════════════════════════════════════════════════════════════════
        # 
        # Using log-price improves numerics:
        # 1. Diffusion coefficient becomes V (not VS²)
        # 2. More uniform resolution across price range
        # 3. Natural for multiplicative processes
        #
        # Grid points: x_i = x_min + i·Δx
        # where Δx = (x_max - x_min)/(N_S - 1)
        # ═══════════════════════════════════════════════════════════════════
        
        self.x_min = np.log(S_min)
        self.x_max = np.log(S_max)
        self.dx = (self.x_max - self.x_min) / (N_S - 1)
        self.x = np.linspace(self.x_min, self.x_max, N_S)
        
        # Price grid (for convenience)
        # S_i = exp(x_i)
        self.S = np.exp(self.x)
        
        # ═══════════════════════════════════════════════════════════════════
        # VARIANCE GRID (uniform)
        # ═══════════════════════════════════════════════════════════════════
        #
        # V_j = V_min + j·ΔV for j = 0, ..., N_V-1
        # where ΔV = (V_max - V_min)/(N_V - 1)
        #
        # Note: V_min should be small positive (not zero)
        # to avoid singularities in √V terms
        # ═══════════════════════════════════════════════════════════════════
        
        self.dV = (V_max - V_min) / (N_V - 1)
        self.V = np.linspace(V_min, V_max, N_V)
        
        # ═══════════════════════════════════════════════════════════════════
        # TIME GRID (backward stepping)
        # ═══════════════════════════════════════════════════════════════════
        #
        # τ = T - t is time-to-maturity
        # We solve from τ = 0 (expiry) to τ = T (today)
        #
        # τ_n = n·Δτ for n = 0, ..., N_t
        # where Δτ = T/N_t
        #
        # At τ = 0: apply terminal payoff condition
        # At τ = T: obtain option value today
        # ═══════════════════════════════════════════════════════════════════
        
        self.dt = T / N_t  # Time step (Δτ)
        self.tau = np.linspace(0, T, N_t + 1)  # τ from 0 to T
        
        # ═══════════════════════════════════════════════════════════════════
        # MESHGRIDS for vectorized operations
        # ═══════════════════════════════════════════════════════════════════
        #
        # Create 2D arrays for S, V, x at each grid point
        # Using 'ij' indexing: first index is S/x, second is V
        # Shape: (N_S, N_V)
        # ═══════════════════════════════════════════════════════════════════
        
        self.S_mesh, self.V_mesh = np.meshgrid(self.S, self.V, indexing='ij')
        self.x_mesh, self.V_mesh_x = np.meshgrid(self.x, self.V, indexing='ij')
        
        # Store grid dimensions
        self.shape = (N_S, N_V)
        
        # Compute stability information
        self._compute_stability_info()
    
    def _compute_stability_info(self):
        """
        Compute stability-related quantities.
        
        Stability Analysis:
        ═══════════════════════════════════════════════════════════════════════
        
        For explicit schemes, the CFL number determines stability:
        
        CFL_x = V_max · Δτ / Δx²  (should be < 0.5 for stability)
        CFL_V = σ_max² · V_max · Δτ / ΔV²  (should be < 0.5)
        
        For implicit schemes (CN, ADI): unconditionally stable
        but accuracy requires reasonable step sizes.
        
        Grid Peclet numbers (for convection-dominated problems):
        Pe_x = |r - q - V/2| · Δx / (V/2)
        Pe_V = |κ(θ - V)| · ΔV / (σ²V/2)
        
        High Peclet (> 2) may cause oscillations.
        ═══════════════════════════════════════════════════════════════════════
        """
        
        # Estimate maximum variance for CFL
        V_max_eff = self.V_max
        
        # CFL numbers (assuming σ ~ 0.3 for estimation)
        sigma_est = 0.3
        self.cfl_x = V_max_eff * self.dt / (self.dx ** 2)
        self.cfl_V = sigma_est**2 * V_max_eff * self.dt / (self.dV ** 2)
        
        # Grid aspect ratio
        self.aspect_ratio = self.dx / self.dV
    
    def find_index(self, S_target: float, V_target: float) -> Tuple[int, int]:
        """
        Find nearest grid indices for given (S, V) point.
        
        Args:
            S_target: Target spot price
            V_target: Target variance
            
        Returns:
            (i, j): Indices such that (S[i], V[j]) is nearest to (S_target, V_target)
        """
        # Find nearest S index
        i = int(np.argmin(np.abs(self.S - S_target)))
        
        # Find nearest V index  
        j = int(np.argmin(np.abs(self.V - V_target)))
        
        return i, j
    
    def interpolate(
        self, 
        values: np.ndarray, 
        S_target: float, 
        V_target: float,
        method: str = 'bilinear'
    ) -> float:
        """
        Interpolate grid values at arbitrary (S, V) point.
        
        Bilinear Interpolation Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Given four corners: f_{i,j}, f_{i+1,j}, f_{i,j+1}, f_{i+1,j+1}
        
        f(x, y) = (1-α)(1-β)f_{i,j} + α(1-β)f_{i+1,j}
                + (1-α)β·f_{i,j+1} + αβ·f_{i+1,j+1}
        
        where:
        α = (x - x_i)/(x_{i+1} - x_i)
        β = (y - y_j)/(y_{j+1} - y_j)
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            values: 2D array of values on grid, shape (N_S, N_V)
            S_target: Target spot price
            V_target: Target variance
            method: 'nearest' or 'bilinear'
            
        Returns:
            Interpolated value
        """
        
        if method == 'nearest':
            i, j = self.find_index(S_target, V_target)
            return values[i, j]
        
        elif method == 'bilinear':
            # Convert to log-price
            x_target = np.log(S_target)
            
            # Find bracketing indices for x
            i = np.searchsorted(self.x, x_target) - 1
            i = max(0, min(i, self.N_S - 2))
            
            # Find bracketing indices for V
            j = np.searchsorted(self.V, V_target) - 1
            j = max(0, min(j, self.N_V - 2))
            
            # Compute interpolation weights
            # α = (x - x_i)/(x_{i+1} - x_i)
            alpha = (x_target - self.x[i]) / (self.x[i+1] - self.x[i])
            alpha = max(0, min(1, alpha))
            
            # β = (V - V_j)/(V_{j+1} - V_j)
            beta = (V_target - self.V[j]) / (self.V[j+1] - self.V[j])
            beta = max(0, min(1, beta))
            
            # Bilinear interpolation
            # f = (1-α)(1-β)f_{00} + α(1-β)f_{10} + (1-α)β·f_{01} + αβ·f_{11}
            f = ((1 - alpha) * (1 - beta) * values[i, j] +
                 alpha * (1 - beta) * values[i+1, j] +
                 (1 - alpha) * beta * values[i, j+1] +
                 alpha * beta * values[i+1, j+1])
            
            return f
        
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def refine(self, factor: int = 2) -> 'Grid':
        """
        Create refined grid with more points.
        
        Args:
            factor: Refinement factor (2 means double points in each dimension)
            
        Returns:
            New Grid with increased resolution
        """
        return Grid(
            S_min=self.S_min,
            S_max=self.S_max,
            N_S=self.N_S * factor,
            V_min=self.V_min,
            V_max=self.V_max,
            N_V=self.N_V * factor,
            T=self.T,
            N_t=self.N_t * factor
        )
    
    def summary(self) -> str:
        """Return grid summary string."""
        return (
            f"Grid Summary:\n"
            f"  S-grid: [{self.S_min:.2f}, {self.S_max:.2f}], N={self.N_S}, Δx={self.dx:.4f}\n"
            f"  V-grid: [{self.V_min:.2e}, {self.V_max:.4f}], N={self.N_V}, ΔV={self.dV:.4f}\n"
            f"  τ-grid: [0, {self.T:.2f}], N={self.N_t}, Δτ={self.dt:.4f}\n"
            f"  CFL numbers: x={self.cfl_x:.2f}, V={self.cfl_V:.2f}"
        )


def create_default_grid(params, T: float, strike: Optional[float] = None) -> Grid:
    """
    Create default grid suitable for given parameters.
    
    Grid Sizing Heuristics:
    ═══════════════════════════════════════════════════════════════════════════
    
    S-bounds: Center around spot, extend to capture tails
    - S_min = 0.2·S₀ (covers deep OTM puts)
    - S_max = 5·S₀ (covers deep OTM calls)
    
    V-bounds: Cover realistic variance range
    - V_min = 10⁻⁸ (essentially zero)
    - V_max = max(5θ, 5V₀, 1.0) (cover high-vol scenarios)
    
    Grid points: Balance accuracy vs computation
    - N_S = 100 (200 for production)
    - N_V = 50 (100 for production)
    - N_t = 100 (scale with T for long maturities)
    
    ═══════════════════════════════════════════════════════════════════════════
    
    Args:
        params: HestonParams object
        T: Time to maturity
        strike: Strike price (optional, for centering grid)
        
    Returns:
        Grid object configured for the parameters
    """
    S0 = params.S0
    K = strike if strike is not None else S0
    
    # S-bounds (logarithmically centered around max(S0, K))
    S_center = max(S0, K)
    S_min = S_center * 0.2
    S_max = S_center * 5.0
    
    # V-bounds
    V_min = 1e-8  # Small positive to avoid singularity
    V_max = max(5 * params.theta, 5 * params.V0, 1.0)
    
    # Grid sizes
    N_S = 100
    N_V = 50
    N_t = max(100, int(100 * T))  # Scale with maturity
    
    return Grid(
        S_min=S_min,
        S_max=S_max,
        N_S=N_S,
        V_min=V_min,
        V_max=V_max,
        N_V=N_V,
        T=T,
        N_t=N_t
    )
