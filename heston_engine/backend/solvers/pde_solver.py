"""
PDE Solver using ADI (Alternating Direction Implicit) Method

═══════════════════════════════════════════════════════════════════════════════
HESTON PDE AND NUMERICAL SOLUTION
═══════════════════════════════════════════════════════════════════════════════

1. HESTON PDE (Backward Kolmogorov Equation):
   ═══════════════════════════════════════════════════════════════════════════
   
   For option value U(S, V, t):
   
   ∂U/∂t + (r-q)S·∂U/∂S + κ(θ-V)·∂U/∂V 
   + (1/2)VS²·∂²U/∂S² + (1/2)σ²V·∂²U/∂V² 
   + ρσVS·∂²U/∂S∂V - rU = 0
   
   Derivation using risk-neutral pricing:
   1. Apply 2D Itô's lemma to U(S_t, V_t, t)
   2. Compute quadratic variations:
      - (dS)² = VS² dt
      - (dV)² = σ²V dt
      - (dS)(dV) = ρσVS dt
   3. Take expectation under risk-neutral measure
   4. Set E[dU] = r·U dt (risk-neutral drift)
   5. Rearrange to get PDE

2. LOG-PRICE TRANSFORMATION (x = ln S):
   ═══════════════════════════════════════════════════════════════════════════
   
   Transform S → x = ln(S) for better numerics:
   
   Original derivatives:
   ∂/∂S = (1/S)·∂/∂x
   ∂²/∂S² = (1/S²)·(∂²/∂x² - ∂/∂x)
   
   Transformed PDE:
   ∂U/∂t + (r-q-V/2)·∂U/∂x + κ(θ-V)·∂U/∂V 
   + (V/2)·∂²U/∂x² + (σ²V/2)·∂²U/∂V² 
   + ρσV·∂²U/∂x∂V - rU = 0
   
   Benefits:
   - Diffusion coefficients V/2 and σ²V/2 depend only on V
   - More uniform resolution across price range
   - Natural for percentage changes

3. OPERATOR SPLITTING (ADI):
   ═══════════════════════════════════════════════════════════════════════════
   
   Split PDE operator into directional components:
   
   L = L_x + L_V + L_xV + L_0
   
   where:
   L_x[U] = (r-q-V/2)·∂U/∂x + (V/2)·∂²U/∂x²
   L_V[U] = κ(θ-V)·∂U/∂V + (σ²V/2)·∂²U/∂V²
   L_xV[U] = ρσV·∂²U/∂x∂V
   L_0[U] = -rU
   
   Full PDE: ∂U/∂τ = L[U]  (τ = T - t, forward in τ)

4. DOUGLAS-RACHFORD ADI SCHEME:
   ═══════════════════════════════════════════════════════════════════════════
   
   Two half-steps per time step:
   
   Step 1 (implicit in x):
   (I - θΔτ·L_x)U* = [I + θΔτ·L_V + Δτ·L_xV + Δτ·L_0]U^n
   
   Step 2 (implicit in V):
   (I - θΔτ·L_V)U^{n+1} = U* - θΔτ·L_V·U^n
   
   where θ = 0.5 for Crank-Nicolson-like accuracy (second order in time)
   
   Key properties:
   - Each step solves tridiagonal systems (Thomas algorithm O(N))
   - Total: O(N_S × N_V) per time step
   - Unconditionally stable for θ ≥ 0.5
   - Second-order accurate in space and time

5. FINITE DIFFERENCE DISCRETIZATION:
   ═══════════════════════════════════════════════════════════════════════════
   
   Central differences for interior points:
   
   ∂U/∂x ≈ (U_{i+1,j} - U_{i-1,j}) / (2Δx)
   
   ∂²U/∂x² ≈ (U_{i+1,j} - 2U_{i,j} + U_{i-1,j}) / Δx²
   
   ∂U/∂V ≈ (U_{i,j+1} - U_{i,j-1}) / (2ΔV)
   
   ∂²U/∂V² ≈ (U_{i,j+1} - 2U_{i,j} + U_{i,j-1}) / ΔV²
   
   Mixed derivative (standard cross stencil):
   ∂²U/∂x∂V ≈ (U_{i+1,j+1} - U_{i+1,j-1} - U_{i-1,j+1} + U_{i-1,j-1}) / (4Δx·ΔV)

6. TRIDIAGONAL SYSTEM (Thomas Algorithm):
   ═══════════════════════════════════════════════════════════════════════════
   
   Solve: a_i·U_{i-1} + b_i·U_i + c_i·U_{i+1} = d_i
   
   Forward elimination:
   c'_i = c_i / (b_i - a_i·c'_{i-1}),  c'_0 = c_0/b_0
   d'_i = (d_i - a_i·d'_{i-1}) / (b_i - a_i·c'_{i-1})
   
   Back substitution:
   U_n = d'_n
   U_i = d'_i - c'_i·U_{i+1}  for i = n-1, ..., 0
   
   O(N) operations, highly efficient

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Tuple, Optional
from backend.core.parameters import HestonParams
from backend.core.grid import Grid
from backend.core.boundaries import BoundaryConditions


class PDESolver:
    """
    Heston PDE solver using ADI (Alternating Direction Implicit) method.
    
    Solves the 2D parabolic PDE backward in time from terminal condition
    to obtain option prices.
    """
    
    def __init__(self, params: HestonParams, grid: Grid):
        """
        Initialize PDE solver.
        
        Args:
            params: Heston model parameters
            grid: Computational grid
        """
        self.p = params  # Parameters
        self.g = grid    # Grid
        self.bc = BoundaryConditions()
        
        # Precompute coefficient arrays for efficiency
        self._precompute_coefficients()
    
    def _precompute_coefficients(self):
        """
        Precompute PDE coefficients on the grid.
        
        Coefficient Arrays:
        ═══════════════════════════════════════════════════════════════════════
        
        For each grid point (i, j), the PDE has coefficients:
        
        L_x coefficients (at variance V_j):
        - First-order: α_j = (r - q - V_j/2)
        - Second-order: β_j = V_j/2
        
        L_V coefficients (at variance V_j):
        - First-order: γ_j = κ(θ - V_j)
        - Second-order: δ_j = σ²V_j/2
        
        L_xV coefficient:
        - Cross-term: η_j = ρσV_j
        
        L_0 coefficient:
        - Discounting: -r
        
        ═══════════════════════════════════════════════════════════════════════
        """
        V = self.g.V  # Variance grid
        
        # L_x coefficients (depend on V)
        # α = r - q - V/2 (drift coefficient in x)
        self.alpha = self.p.r - self.p.q - V / 2
        
        # β = V/2 (diffusion coefficient in x)
        self.beta = V / 2
        
        # L_V coefficients
        # γ = κ(θ - V) (drift coefficient in V)
        self.gamma = self.p.kappa * (self.p.theta - V)
        
        # δ = σ²V/2 (diffusion coefficient in V)
        self.delta = self.p.sigma**2 * V / 2
        
        # L_xV coefficient
        # η = ρσV (mixed derivative coefficient)
        self.eta = self.p.rho * self.p.sigma * V
    
    def solve_european_call(self, K: float) -> np.ndarray:
        """
        Solve Heston PDE for European call option.
        
        Algorithm:
        ═══════════════════════════════════════════════════════════════════════
        
        1. Initialize: U(x, V, T) = max(e^x - K, 0)  (terminal condition)
        
        2. For each time step n = 0, 1, ..., N_t - 1:
           a. Compute τ = (n+1)·Δτ (time to maturity after step)
           b. ADI Step 1: Solve implicit in x-direction
           c. ADI Step 2: Solve implicit in V-direction
           d. Apply boundary conditions
        
        3. Return: U(x, V, 0) = option prices at t = 0
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            K: Strike price
            
        Returns:
            U: Option value grid, shape (N_S, N_V)
        """
        return self._solve_european(K, option_type='call')
    
    def solve_european_put(self, K: float) -> np.ndarray:
        """
        Solve Heston PDE for European put option.
        
        Args:
            K: Strike price
            
        Returns:
            U: Option value grid, shape (N_S, N_V)
        """
        return self._solve_european(K, option_type='put')
    
    def _solve_european(self, K: float, option_type: str = 'call') -> np.ndarray:
        """
        Core solver for European options.
        
        Implementation Details:
        ═══════════════════════════════════════════════════════════════════════
        
        The ADI scheme proceeds as follows:
        
        At each time step, we go from U^n to U^{n+1}:
        
        1. Compute RHS including explicit V and cross-derivative terms
        2. Solve tridiagonal system in x for each j (implicit in x)
        3. Solve tridiagonal system in V for each i (implicit in V)
        4. Apply Dirichlet boundary conditions
        
        ═══════════════════════════════════════════════════════════════════════
        """
        # Get grid dimensions
        N_S, N_V = self.g.N_S, self.g.N_V
        dx, dV, dt = self.g.dx, self.g.dV, self.g.dt
        
        # ADI parameter (θ = 0.5 for Crank-Nicolson)
        theta = 0.5
        
        # Initialize solution array with terminal condition
        # U(S, V, T) = payoff(S)
        U = np.zeros((N_S, N_V))
        
        if option_type == 'call':
            for i in range(N_S):
                U[i, :] = self.bc.call_payoff(self.g.S[i], K)
        else:
            for i in range(N_S):
                U[i, :] = self.bc.put_payoff(self.g.S[i], K)
        
        # Time stepping (backward from T to 0)
        for n in range(self.g.N_t):
            tau = self.g.tau[n + 1]  # Time to maturity after this step
            
            # ═══════════════════════════════════════════════════════════════
            # ADI STEP 1: Implicit in x-direction
            # ═══════════════════════════════════════════════════════════════
            # 
            # Solve: (I - θΔτ·L_x)U* = [I + θΔτ·L_V + Δτ·L_xV - Δτ·r]U^n
            #
            # For each variance level j, solve tridiagonal system in i
            # ═══════════════════════════════════════════════════════════════
            
            U_star = np.zeros_like(U)
            
            for j in range(N_V):
                V_j = self.g.V[j]
                
                # Coefficients for x-direction operator at this V_j
                # L_x[U] = α·∂U/∂x + β·∂²U/∂x²
                # where α = r - q - V/2, β = V/2
                
                alpha_j = self.alpha[j]  # r - q - V_j/2
                beta_j = self.beta[j]    # V_j/2
                
                # Build tridiagonal matrix coefficients for (I - θΔτ·L_x)
                # 
                # L_x discretized:
                # L_x[U_i] = α(U_{i+1}-U_{i-1})/(2Δx) + β(U_{i+1}-2U_i+U_{i-1})/Δx²
                #
                # = (-β/Δx² - α/(2Δx))U_{i-1} + (2β/Δx²)U_i + (-β/Δx² + α/(2Δx))U_{i+1}
                #
                # Tridiagonal entries for (I - θΔτ·L_x):
                # a_i = θΔτ(β/Δx² + α/(2Δx))  [lower diagonal]
                # b_i = 1 - θΔτ(−2β/Δx²)      [diagonal]
                # c_i = θΔτ(β/Δx² - α/(2Δx))  [upper diagonal]
                
                coef_diff = beta_j / (dx**2)    # β/Δx²
                coef_conv = alpha_j / (2 * dx)   # α/(2Δx)
                
                a_coef = theta * dt * (coef_diff + coef_conv)
                b_coef = 1 + theta * dt * 2 * coef_diff
                c_coef = theta * dt * (coef_diff - coef_conv)
                
                # Build RHS: [I + θΔτ·L_V + Δτ·L_xV - Δτ·r]U^n
                rhs = np.zeros(N_S)
                
                for i in range(1, N_S - 1):
                    # Explicit L_V contribution
                    if j > 0 and j < N_V - 1:
                        gamma_j = self.gamma[j]
                        delta_j = self.delta[j]
                        
                        # L_V[U] = γ·∂U/∂V + δ·∂²U/∂V²
                        dU_dV = (U[i, j+1] - U[i, j-1]) / (2 * dV)
                        d2U_dV2 = (U[i, j+1] - 2*U[i, j] + U[i, j-1]) / (dV**2)
                        L_V_U = gamma_j * dU_dV + delta_j * d2U_dV2
                        
                        rhs[i] += theta * dt * L_V_U
                    
                    # Explicit L_xV contribution (mixed derivative)
                    if i > 0 and i < N_S - 1 and j > 0 and j < N_V - 1:
                        eta_j = self.eta[j]
                        
                        # L_xV[U] = η·∂²U/∂x∂V
                        # ∂²U/∂x∂V ≈ (U_{i+1,j+1} - U_{i+1,j-1} - U_{i-1,j+1} + U_{i-1,j-1})/(4Δx·ΔV)
                        d2U_dxdV = (U[i+1, j+1] - U[i+1, j-1] - U[i-1, j+1] + U[i-1, j-1]) / (4 * dx * dV)
                        L_xV_U = eta_j * d2U_dxdV
                        
                        rhs[i] += dt * L_xV_U
                    
                    # Discounting term
                    rhs[i] -= dt * self.p.r * U[i, j]
                    
                    # Add current value
                    rhs[i] += U[i, j]
                    
                    # Explicit x-direction contribution (for RHS)
                    if i > 0 and i < N_S - 1:
                        dU_dx = (U[i+1, j] - U[i-1, j]) / (2 * dx)
                        d2U_dx2 = (U[i+1, j] - 2*U[i, j] + U[i-1, j]) / (dx**2)
                        L_x_U_explicit = alpha_j * dU_dx + beta_j * d2U_dx2
                        
                        rhs[i] += (1 - theta) * dt * L_x_U_explicit
                
                # Boundary conditions for RHS
                if option_type == 'call':
                    rhs[0] = 0  # S_min boundary
                    rhs[-1] = max(self.g.S[-1] * np.exp(-self.p.q * tau) - 
                                 K * np.exp(-self.p.r * tau), 0)
                else:
                    rhs[0] = K * np.exp(-self.p.r * tau)
                    rhs[-1] = 0
                
                # Solve tridiagonal system
                U_star[:, j] = self._solve_tridiagonal_x(a_coef, b_coef, c_coef, rhs)
            
            # ═══════════════════════════════════════════════════════════════
            # ADI STEP 2: Implicit in V-direction
            # ═══════════════════════════════════════════════════════════════
            #
            # Solve: (I - θΔτ·L_V)U^{n+1} = U* - θΔτ·L_V·U^n
            #
            # For each S level i, solve tridiagonal system in j
            # ═══════════════════════════════════════════════════════════════
            
            U_new = np.zeros_like(U)
            
            for i in range(N_S):
                S_i = self.g.S[i]
                
                # Build tridiagonal system for V direction
                a_vec = np.zeros(N_V)
                b_vec = np.ones(N_V)
                c_vec = np.zeros(N_V)
                rhs = np.copy(U_star[i, :])
                
                for j in range(1, N_V - 1):
                    gamma_j = self.gamma[j]
                    delta_j = self.delta[j]
                    
                    # Coefficients for L_V at V_j
                    # L_V[U] = γ·∂U/∂V + δ·∂²U/∂V²
                    #
                    # Discretized:
                    # L_V[U_j] = γ(U_{j+1}-U_{j-1})/(2ΔV) + δ(U_{j+1}-2U_j+U_{j-1})/ΔV²
                    #
                    # Tridiagonal for (I - θΔτ·L_V):
                    
                    coef_diff_V = delta_j / (dV**2)
                    coef_conv_V = gamma_j / (2 * dV)
                    
                    a_vec[j] = theta * dt * (coef_diff_V + coef_conv_V)
                    b_vec[j] = 1 + theta * dt * 2 * coef_diff_V
                    c_vec[j] = theta * dt * (coef_diff_V - coef_conv_V)
                    
                    # RHS correction: - θΔτ·L_V·U^n
                    if j > 0 and j < N_V - 1:
                        dU_dV_old = (U[i, j+1] - U[i, j-1]) / (2 * dV)
                        d2U_dV2_old = (U[i, j+1] - 2*U[i, j] + U[i, j-1]) / (dV**2)
                        L_V_U_old = gamma_j * dU_dV_old + delta_j * d2U_dV2_old
                        
                        rhs[j] -= theta * dt * L_V_U_old
                
                # V boundary conditions
                # V = 0: Use Black-Scholes with σ = √θ
                sigma_eff = np.sqrt(self.p.theta)
                if option_type == 'call':
                    rhs[0] = self.bc.black_scholes_call(S_i, K, tau, self.p.r, self.p.q, sigma_eff)
                    rhs[-1] = S_i * np.exp(-self.p.q * tau)
                else:
                    rhs[0] = self.bc.black_scholes_put(S_i, K, tau, self.p.r, self.p.q, sigma_eff)
                    rhs[-1] = K * np.exp(-self.p.r * tau)
                
                b_vec[0] = 1  # Dirichlet at V_min
                b_vec[-1] = 1  # Dirichlet at V_max
                
                # Solve tridiagonal system
                U_new[i, :] = self._solve_tridiagonal_V(a_vec, b_vec, c_vec, rhs)
            
            # Apply boundary conditions
            U = self._apply_boundaries(U_new, K, tau, option_type)
        
        return U
    
    def _solve_tridiagonal_x(
        self, 
        a: float, 
        b: float, 
        c: float, 
        d: np.ndarray
    ) -> np.ndarray:
        """
        Solve tridiagonal system with constant coefficients using Thomas algorithm.
        
        Thomas Algorithm:
        ═══════════════════════════════════════════════════════════════════════
        
        System: a_i·x_{i-1} + b_i·x_i + c_i·x_{i+1} = d_i
        
        Forward elimination (eliminate lower diagonal):
        w = a / b_{prev}
        b'_i = b - w·c_{prev}
        d'_i = d_i - w·d'_{prev}
        
        Back substitution:
        x_n = d'_n / b'_n
        x_i = (d'_i - c·x_{i+1}) / b'_i
        
        O(n) operations, numerically stable for diagonally dominant matrices
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            a: Lower diagonal coefficient (constant)
            b: Main diagonal coefficient (constant)
            c: Upper diagonal coefficient (constant)
            d: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        n = len(d)
        x = np.zeros(n)
        
        # Work arrays
        b_mod = np.zeros(n)
        d_mod = np.zeros(n)
        
        # Initialize
        b_mod[0] = b
        d_mod[0] = d[0]
        
        # Forward elimination
        for i in range(1, n):
            w = a / b_mod[i-1]
            b_mod[i] = b - w * c
            d_mod[i] = d[i] - w * d_mod[i-1]
        
        # Back substitution
        x[-1] = d_mod[-1] / b_mod[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (d_mod[i] - c * x[i+1]) / b_mod[i]
        
        return x
    
    def _solve_tridiagonal_V(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray
    ) -> np.ndarray:
        """
        Solve tridiagonal system with variable coefficients.
        
        Args:
            a: Lower diagonal coefficients (array)
            b: Main diagonal coefficients (array)
            c: Upper diagonal coefficients (array)
            d: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        n = len(d)
        x = np.zeros(n)
        
        # Work arrays
        c_mod = np.zeros(n)
        d_mod = np.zeros(n)
        
        # Initialize
        c_mod[0] = c[0] / b[0] if abs(b[0]) > 1e-15 else 0
        d_mod[0] = d[0] / b[0] if abs(b[0]) > 1e-15 else d[0]
        
        # Forward elimination
        for i in range(1, n):
            denom = b[i] - a[i] * c_mod[i-1]
            if abs(denom) < 1e-15:
                denom = 1e-15
            c_mod[i] = c[i] / denom if i < n - 1 else 0
            d_mod[i] = (d[i] - a[i] * d_mod[i-1]) / denom
        
        # Back substitution
        x[-1] = d_mod[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_mod[i] - c_mod[i] * x[i+1]
        
        return x
    
    def _apply_boundaries(
        self, 
        U: np.ndarray, 
        K: float, 
        tau: float,
        option_type: str
    ) -> np.ndarray:
        """
        Apply boundary conditions to solution grid.
        
        Boundary Conditions Applied:
        ═══════════════════════════════════════════════════════════════════════
        
        S = S_min:
          Call: U = 0
          Put: U = K·e^{-rτ}
        
        S = S_max:
          Call: U = S_max·e^{-qτ} - K·e^{-rτ}
          Put: U = 0
        
        V = V_min:
          Use Black-Scholes with σ = √θ
        
        V = V_max:
          Call: U = S·e^{-qτ}
          Put: U = K·e^{-rτ}
        
        ═══════════════════════════════════════════════════════════════════════
        """
        if option_type == 'call':
            # S = 0 boundary
            U[0, :] = 0.0
            
            # S = S_max boundary
            U[-1, :] = max(self.g.S[-1] * np.exp(-self.p.q * tau) - 
                          K * np.exp(-self.p.r * tau), 0.0)
        else:
            # S = 0 boundary
            U[0, :] = K * np.exp(-self.p.r * tau)
            
            # S = S_max boundary
            U[-1, :] = 0.0
        
        return U
    
    def get_price(self, U_grid: np.ndarray, S: float, V: float) -> float:
        """
        Interpolate option price at specific (S, V) from solution grid.
        
        Args:
            U_grid: Solution grid from solve_european_*
            S: Spot price
            V: Variance
            
        Returns:
            Interpolated option price
        """
        return self.g.interpolate(U_grid, S, V, method='bilinear')
    
    def compute_delta(self, U_grid: np.ndarray, S: float, V: float) -> float:
        """
        Compute Delta (∂U/∂S) via finite difference.
        
        Delta Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Δ = ∂U/∂S ≈ (U(S+h) - U(S-h)) / (2h)
        
        Or using x = ln(S):
        Δ = (1/S) · ∂U/∂x ≈ (1/S) · (U_{i+1} - U_{i-1}) / (2Δx)
        
        ═══════════════════════════════════════════════════════════════════════
        """
        i, j = self.g.find_index(S, V)
        
        # Ensure we're not at boundary
        i = max(1, min(i, self.g.N_S - 2))
        
        # Finite difference
        dU_dx = (U_grid[i+1, j] - U_grid[i-1, j]) / (2 * self.g.dx)
        
        # Delta = (1/S) · dU/dx
        return dU_dx / S
    
    def compute_gamma(self, U_grid: np.ndarray, S: float, V: float) -> float:
        """
        Compute Gamma (∂²U/∂S²) via finite difference.
        
        Gamma Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Γ = ∂²U/∂S² = (1/S²)(∂²U/∂x² - ∂U/∂x)
            ≈ (1/S²)[(U_{i+1} - 2U_i + U_{i-1})/Δx² - (U_{i+1} - U_{i-1})/(2Δx)]
        
        ═══════════════════════════════════════════════════════════════════════
        """
        i, j = self.g.find_index(S, V)
        i = max(1, min(i, self.g.N_S - 2))
        
        dx = self.g.dx
        
        d2U_dx2 = (U_grid[i+1, j] - 2*U_grid[i, j] + U_grid[i-1, j]) / (dx**2)
        dU_dx = (U_grid[i+1, j] - U_grid[i-1, j]) / (2 * dx)
        
        # Gamma = (1/S²)(d²U/dx² - dU/dx)
        return (d2U_dx2 - dU_dx) / (S**2)
    
    def compute_vega(self, U_grid: np.ndarray, S: float, V: float) -> float:
        """
        Compute Vega (∂U/∂V) via finite difference.
        
        Note: This is vega with respect to variance V, not volatility σ.
        For volatility vega, multiply by 2√V.
        
        Vega Formula:
        ═══════════════════════════════════════════════════════════════════════
        
        Vega_V = ∂U/∂V ≈ (U_{j+1} - U_{j-1}) / (2ΔV)
        
        Vega_σ = ∂U/∂σ = (∂U/∂V)·(∂V/∂σ) = 2σ·(∂U/∂V)
        
        ═══════════════════════════════════════════════════════════════════════
        """
        i, j = self.g.find_index(S, V)
        j = max(1, min(j, self.g.N_V - 2))
        
        # dU/dV
        vega_V = (U_grid[i, j+1] - U_grid[i, j-1]) / (2 * self.g.dV)
        
        return vega_V
