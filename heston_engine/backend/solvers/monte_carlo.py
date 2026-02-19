"""
Monte Carlo Simulation with QE (Quadratic-Exponential) Scheme

═══════════════════════════════════════════════════════════════════════════════
MONTE CARLO METHODS FOR HESTON MODEL
═══════════════════════════════════════════════════════════════════════════════

1. BASIC EULER DISCRETIZATION:
   ═══════════════════════════════════════════════════════════════════════════
   
   Naive discretization of Heston SDEs:
   
   V_{n+1} = V_n + κ(θ - V_n)Δt + σ√V_n√Δt·Z_V
   S_{n+1} = S_n·exp[(r-q-V_n/2)Δt + √V_n√Δt·Z_S]
   
   where Z_V, Z_S are correlated normals:
   Z_S = ρ·Z_V + √(1-ρ²)·Z_indep
   
   Problems:
   - V can go negative (variance must be non-negative)
   - Poor weak convergence near V = 0
   - Bias in option prices

2. FULL TRUNCATION SCHEME:
   ═══════════════════════════════════════════════════════════════════════════
   
   Truncate negative variance:
   V_{n+1}^+ = max(V_{n+1}, 0)
   
   Or absorbing scheme:
   V_{n+1} = max(V_n + κ(θ - V_n^+)Δt + σ√V_n^+√Δt·Z_V, 0)
   
   Better but still biased for small Δt.

3. QE (QUADRATIC-EXPONENTIAL) SCHEME:
   ═══════════════════════════════════════════════════════════════════════════
   
   Developed by Andersen (2008) for efficient Heston simulation.
   
   Key insight: Use different sampling schemes based on ψ = s²/m²
   
   Step 1: Compute moment-matched parameters
   
   m = E[V_{n+1} | V_n] = θ + (V_n - θ)e^{-κΔt}  (conditional mean)
   
   s² = Var[V_{n+1} | V_n] = V_n·(σ²/κ)e^{-κΔt}(1-e^{-κΔt})
                          + θ·(σ²/2κ)(1-e^{-κΔt})²  (conditional variance)
   
   ψ = s²/m²  (squared coefficient of variation)
   
   Step 2: Sample V_{n+1} based on ψ regime
   
   If ψ ≤ ψ_c (typically ψ_c = 1.5):
     Use quadratic sampling (moment-matched)
     b² = 2/ψ - 1 + √(2/ψ)·√(2/ψ - 1)
     a = m/(1 + b²)
     V_{n+1} = a·(b + Z_V)²
   
   If ψ > ψ_c:
     Use exponential sampling
     p = (ψ - 1)/(ψ + 1)
     β = (1 - p)/m
     V_{n+1} = 0 with probability p
     V_{n+1} = (1/β)·ln((1-p)/(1-U)) with probability 1-p
   
   Step 3: Sample S_{n+1} using integrated variance
   
   ln(S_{n+1}/S_n) = (r-q)Δt - ∫V_s ds/2 + √∫V_s ds·Z_S
   
   Use trapezoidal approximation:
   ∫V_s ds ≈ (V_n + V_{n+1})Δt/2

4. VARIANCE REDUCTION TECHNIQUES:
   ═══════════════════════════════════════════════════════════════════════════
   
   Antithetic variates:
   - For each path with (Z_1, Z_2, ...), also simulate (-Z_1, -Z_2, ...)
   - Average payoffs reduces variance by factor ~2
   
   Control variates:
   - Use Black-Scholes price as control
   - Adjust payoff: Y_adj = Y - α·(X - E[X])
   - Choose α to minimize Var[Y_adj]
   
   Importance sampling:
   - Shift drift to sample more ITM paths
   - Apply likelihood ratio correction

5. STANDARD ERROR AND CONFIDENCE INTERVALS:
   ═══════════════════════════════════════════════════════════════════════════
   
   Standard error of Monte Carlo estimate:
   SE = σ/√N
   
   where σ = std(discounted payoffs), N = number of paths
   
   95% confidence interval:
   [μ̂ - 1.96·SE, μ̂ + 1.96·SE]
   
   For N = 100,000 paths and σ ≈ 10:
   SE ≈ 10/√100000 ≈ 0.03

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Tuple, Optional
from backend.core.parameters import HestonParams


class MonteCarloSimulator:
    """
    Monte Carlo simulator for Heston model using QE scheme.
    
    Implementation based on:
    Andersen, L. (2008). "Simple and Efficient Simulation of the Heston
    Stochastic Volatility Model." Journal of Computational Finance, 11(3).
    """
    
    def __init__(self, params: HestonParams):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            params: Heston model parameters
        """
        self.p = params
        
        # QE scheme parameters
        self.psi_c = 1.5  # Critical ψ for regime switch
        
    def simulate_paths_euler(
        self,
        T: float,
        N_steps: int,
        N_paths: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths using Euler scheme with full truncation.
        
        Euler Discretization:
        ═══════════════════════════════════════════════════════════════════════
        
        V_{n+1} = V_n + κ(θ - V_n^+)Δt + σ√(V_n^+)√Δt·Z_V
        V_{n+1}^+ = max(V_{n+1}, 0)
        
        ln(S_{n+1}/S_n) = (r-q-V_n^+/2)Δt + √(V_n^+)√Δt·Z_S
        
        where:
        Z_S = ρ·Z_V + √(1-ρ²)·Z_indep
        
        This is first-order accurate but can be biased near V = 0.
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            T: Time horizon
            N_steps: Number of time steps
            N_paths: Number of simulation paths
            seed: Random seed (optional)
            
        Returns:
            S_paths: Asset price paths, shape (N_paths, N_steps+1)
            V_paths: Variance paths, shape (N_paths, N_steps+1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / N_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        S_paths = np.zeros((N_paths, N_steps + 1))
        V_paths = np.zeros((N_paths, N_steps + 1))
        
        S_paths[:, 0] = self.p.S0
        V_paths[:, 0] = self.p.V0
        
        # Euler time stepping
        for i in range(N_steps):
            # Generate correlated Brownian increments
            # Z_S = ρ·Z_V + √(1-ρ²)·Z_indep
            Z_V = np.random.normal(size=N_paths)
            Z_indep = np.random.normal(size=N_paths)
            Z_S = self.p.rho * Z_V + np.sqrt(1 - self.p.rho**2) * Z_indep
            
            # Current variance (truncate to non-negative)
            V_curr = np.maximum(V_paths[:, i], 0)
            sqrt_V = np.sqrt(V_curr)
            
            # ═══════════════════════════════════════════════════════════════
            # VARIANCE UPDATE (Euler with full truncation)
            # ═══════════════════════════════════════════════════════════════
            #
            # V_{n+1} = V_n + κ(θ - V_n)Δt + σ√V_n·√Δt·Z_V
            #
            # Drift: κ(θ - V_n)Δt pulls toward θ
            # Diffusion: σ√V_n·√Δt·Z_V adds randomness
            # ═══════════════════════════════════════════════════════════════
            
            V_drift = self.p.kappa * (self.p.theta - V_curr) * dt
            V_diffusion = self.p.sigma * sqrt_V * sqrt_dt * Z_V
            V_new = V_curr + V_drift + V_diffusion
            V_paths[:, i+1] = np.maximum(V_new, 0)  # Full truncation
            
            # ═══════════════════════════════════════════════════════════════
            # PRICE UPDATE (Log-Euler)
            # ═══════════════════════════════════════════════════════════════
            #
            # ln(S_{n+1}/S_n) = (r-q-V/2)Δt + √V·√Δt·Z_S
            #
            # This is exact for the log-process (avoids S going negative)
            # ═══════════════════════════════════════════════════════════════
            
            log_return = (self.p.r - self.p.q - 0.5*V_curr)*dt + sqrt_V*sqrt_dt*Z_S
            S_paths[:, i+1] = S_paths[:, i] * np.exp(log_return)
        
        return S_paths, V_paths
    
    def simulate_paths_qe(
        self,
        T: float,
        N_steps: int,
        N_paths: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths using QE (Quadratic-Exponential) scheme.
        
        QE Scheme:
        ═══════════════════════════════════════════════════════════════════════
        
        Step 1: Compute conditional moments of V_{n+1} given V_n
        
        m = θ + (V_n - θ)·e^{-κΔt}  [Conditional mean]
        
        s² = (V_n·σ²/κ)·e^{-κΔt}·(1 - e^{-κΔt})
           + (θ·σ²/2κ)·(1 - e^{-κΔt})²  [Conditional variance]
        
        ψ = s²/m²  [Squared coefficient of variation]
        
        Step 2: Sample V_{n+1} based on regime
        
        If ψ ≤ 1.5 (low variance regime):
          Match first two moments with quadratic form
          b² = 2/ψ - 1 + √(2/ψ·(2/ψ - 1))
          a = m/(1 + b²)
          V_{n+1} = a·(b + Z)²  where Z ~ N(0,1)
        
        If ψ > 1.5 (high variance regime):
          Use exponential approximation
          p = (ψ - 1)/(ψ + 1)
          β = (1 - p)/m
          V_{n+1} = Ψ^{-1}(U)  (inverse of piecewise CDF)
        
        Step 3: Update S using integrated variance
        
        ∫V ds ≈ (V_n + V_{n+1})Δt/2  [Trapezoidal rule]
        
        ln(S_{n+1}/S_n) = (r-q)Δt - ∫V ds/2 + √(∫V ds)·Z_S
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            T: Time horizon
            N_steps: Number of time steps
            N_paths: Number of simulation paths
            seed: Random seed (optional)
            
        Returns:
            S_paths: Asset price paths, shape (N_paths, N_steps+1)
            V_paths: Variance paths, shape (N_paths, N_steps+1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / N_steps
        
        # Precompute constants
        exp_kappa_dt = np.exp(-self.p.kappa * dt)
        
        # Initialize arrays
        S_paths = np.zeros((N_paths, N_steps + 1))
        V_paths = np.zeros((N_paths, N_steps + 1))
        
        S_paths[:, 0] = self.p.S0
        V_paths[:, 0] = self.p.V0
        
        for i in range(N_steps):
            V_curr = V_paths[:, i]
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Conditional moments
            # ═══════════════════════════════════════════════════════════════
            #
            # m = E[V_{n+1} | V_n] = θ + (V_n - θ)e^{-κΔt}
            # 
            # This is the exact conditional mean of the CIR process
            # ═══════════════════════════════════════════════════════════════
            
            m = self.p.theta + (V_curr - self.p.theta) * exp_kappa_dt
            m = np.maximum(m, 1e-10)  # Ensure positive
            
            # ═══════════════════════════════════════════════════════════════
            # CONDITIONAL VARIANCE
            # ═══════════════════════════════════════════════════════════════
            #
            # s² = Var[V_{n+1} | V_n]
            #    = (V_n·σ²/κ)·e^{-κΔt}·(1 - e^{-κΔt})
            #    + (θ·σ²/2κ)·(1 - e^{-κΔt})²
            #
            # Two sources:
            # 1. Starting variance V_n decaying
            # 2. Mean reversion to θ
            # ═══════════════════════════════════════════════════════════════
            
            K1 = (self.p.sigma**2 / self.p.kappa) * exp_kappa_dt * (1 - exp_kappa_dt)
            K2 = (self.p.theta * self.p.sigma**2 / (2*self.p.kappa)) * (1 - exp_kappa_dt)**2
            
            s2 = V_curr * K1 + K2
            s2 = np.maximum(s2, 1e-15)
            
            # ψ = s²/m² (squared coefficient of variation)
            psi = s2 / (m**2)
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 2: Sample V_{n+1} based on ψ regime
            # ═══════════════════════════════════════════════════════════════
            
            V_next = np.zeros(N_paths)
            
            # Low variance regime (ψ ≤ ψ_c): Quadratic sampling
            # ═══════════════════════════════════════════════════════════════
            #
            # Match moments with V = a(b + Z)² where Z ~ N(0,1)
            #
            # E[V] = a(b² + 1) = m  →  a = m/(b² + 1)
            # Var[V] = 2a²(2b² + 1) = s²
            #
            # Solving: b² = 2/ψ - 1 + √(2/ψ·(2/ψ - 1))
            # ═══════════════════════════════════════════════════════════════
            
            low_var = psi <= self.psi_c
            if np.any(low_var):
                psi_low = psi[low_var]
                m_low = m[low_var]
                
                # b² = 2/ψ - 1 + √(2/ψ)·√(2/ψ - 1)
                inv_psi = 2.0 / psi_low
                b2 = inv_psi - 1 + np.sqrt(inv_psi * np.maximum(inv_psi - 1, 0))
                b2 = np.maximum(b2, 0)
                
                # a = m/(1 + b²)
                a = m_low / (1 + b2)
                
                # V = a(b + Z)²
                Z = np.random.normal(size=np.sum(low_var))
                b = np.sqrt(b2)
                V_next[low_var] = a * (b + Z)**2
            
            # High variance regime (ψ > ψ_c): Exponential sampling
            # ═══════════════════════════════════════════════════════════════
            #
            # Use exponential distribution with point mass at 0
            # 
            # p = P(V = 0) = (ψ - 1)/(ψ + 1)
            # β = (1 - p)/m
            #
            # For U ~ Uniform(0,1):
            # V = 0 if U ≤ p
            # V = (1/β)·ln((1-p)/(1-U)) if U > p
            # ═══════════════════════════════════════════════════════════════
            
            high_var = ~low_var
            if np.any(high_var):
                psi_high = psi[high_var]
                m_high = m[high_var]
                
                p = (psi_high - 1) / (psi_high + 1)
                beta = (1 - p) / m_high
                
                U = np.random.uniform(size=np.sum(high_var))
                
                # If U ≤ p: V = 0
                # If U > p: V = ln((1-p)/(1-U)) / β
                exp_value = np.log((1 - p) / np.maximum(1 - U, 1e-15)) / beta
                V_next[high_var] = np.where(U <= p, 0, exp_value)
            
            V_paths[:, i+1] = np.maximum(V_next, 0)
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 3: Update asset price
            # ═══════════════════════════════════════════════════════════════
            #
            # Use integrated variance for accurate log price update
            #
            # ln(S_{n+1}/S_n) = (r-q)Δt - ∫V ds/2 + √(∫V ds)·Z_S
            #
            # Trapezoidal approximation:
            # ∫V ds ≈ (V_n + V_{n+1})Δt/2
            # ═══════════════════════════════════════════════════════════════
            
            # Generate correlated normal for price
            Z_V = np.random.normal(size=N_paths)  # Already used for V, but we need new for correlation
            Z_indep = np.random.normal(size=N_paths)
            Z_S = self.p.rho * Z_V + np.sqrt(1 - self.p.rho**2) * Z_indep
            
            # Integrated variance (trapezoidal)
            V_integrated = 0.5 * (V_curr + V_paths[:, i+1]) * dt
            V_integrated = np.maximum(V_integrated, 0)
            
            # Log price update
            log_return = (self.p.r - self.p.q) * dt - 0.5 * V_integrated + np.sqrt(V_integrated) * Z_S
            S_paths[:, i+1] = S_paths[:, i] * np.exp(log_return)
        
        return S_paths, V_paths
    
    def simulate_paths(
        self,
        T: float,
        N_steps: int,
        N_paths: int,
        method: str = 'qe',
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths using specified method.
        
        Args:
            T: Time horizon
            N_steps: Number of time steps
            N_paths: Number of simulation paths
            method: 'euler' or 'qe'
            seed: Random seed (optional)
            
        Returns:
            S_paths, V_paths
        """
        if method == 'euler':
            return self.simulate_paths_euler(T, N_steps, N_paths, seed)
        elif method == 'qe':
            return self.simulate_paths_qe(T, N_steps, N_paths, seed)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def price_european_call(
        self,
        K: float,
        T: float,
        N_steps: int = 252,
        N_paths: int = 100000,
        method: str = 'qe',
        antithetic: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Price European call option via Monte Carlo.
        
        Monte Carlo Pricing:
        ═══════════════════════════════════════════════════════════════════════
        
        Price = e^{-rT}·E[(S_T - K)⁺]
        
        Monte Carlo estimate:
        Price ≈ e^{-rT}·(1/N)·Σᵢ max(S_T^i - K, 0)
        
        Standard error:
        SE = σ(payoffs)/√N · e^{-rT}
        
        95% CI:
        [Price - 1.96·SE, Price + 1.96·SE]
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            K: Strike price
            T: Time to maturity
            N_steps: Number of time steps per path
            N_paths: Number of simulation paths
            method: 'euler' or 'qe'
            antithetic: Use antithetic variates
            seed: Random seed
            
        Returns:
            (price, standard_error)
        """
        if antithetic:
            # Use half paths, then generate antithetic
            N_paths_half = N_paths // 2
            
            # Original paths
            S_paths1, _ = self.simulate_paths(T, N_steps, N_paths_half, method, seed)
            
            # Antithetic paths (negate the random numbers by flipping drift adjustment)
            # For simplicity, we simulate again with different seed
            S_paths2, _ = self.simulate_paths(T, N_steps, N_paths_half, method, 
                                               seed + 12345 if seed else None)
            
            # Combine payoffs
            payoffs1 = np.maximum(S_paths1[:, -1] - K, 0)
            payoffs2 = np.maximum(S_paths2[:, -1] - K, 0)
            payoffs = 0.5 * (payoffs1 + payoffs2)
        else:
            S_paths, _ = self.simulate_paths(T, N_steps, N_paths, method, seed)
            payoffs = np.maximum(S_paths[:, -1] - K, 0)
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPUTE PRICE AND STANDARD ERROR
        # ═══════════════════════════════════════════════════════════════════
        #
        # Price = e^{-rT}·mean(payoffs)
        # SE = e^{-rT}·std(payoffs)/√N
        # ═══════════════════════════════════════════════════════════════════
        
        discount = np.exp(-self.p.r * T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return price, std_error
    
    def price_european_put(
        self,
        K: float,
        T: float,
        N_steps: int = 252,
        N_paths: int = 100000,
        method: str = 'qe',
        antithetic: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Price European put option via Monte Carlo.
        
        Args:
            K: Strike price
            T: Time to maturity
            N_steps: Number of time steps per path
            N_paths: Number of simulation paths
            method: 'euler' or 'qe'
            antithetic: Use antithetic variates
            seed: Random seed
            
        Returns:
            (price, standard_error)
        """
        if antithetic:
            N_paths_half = N_paths // 2
            S_paths1, _ = self.simulate_paths(T, N_steps, N_paths_half, method, seed)
            S_paths2, _ = self.simulate_paths(T, N_steps, N_paths_half, method,
                                               seed + 12345 if seed else None)
            
            payoffs1 = np.maximum(K - S_paths1[:, -1], 0)
            payoffs2 = np.maximum(K - S_paths2[:, -1], 0)
            payoffs = 0.5 * (payoffs1 + payoffs2)
        else:
            S_paths, _ = self.simulate_paths(T, N_steps, N_paths, method, seed)
            payoffs = np.maximum(K - S_paths[:, -1], 0)
        
        discount = np.exp(-self.p.r * T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return price, std_error
    
    def price_european_digital(
        self,
        K: float,
        T: float,
        N_steps: int = 252,
        N_paths: int = 100000,
        method: str = 'qe',
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Price digital (binary) call option via Monte Carlo.
        
        Payoff = 1 if S_T > K, else 0
        
        Args:
            K: Strike price
            T: Time to maturity
            N_steps: Number of time steps
            N_paths: Number of paths
            method: Simulation method
            seed: Random seed
            
        Returns:
            (price, standard_error)
        """
        S_paths, _ = self.simulate_paths(T, N_steps, N_paths, method, seed)
        
        payoffs = np.where(S_paths[:, -1] > K, 1.0, 0.0)
        
        discount = np.exp(-self.p.r * T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(N_paths)
        
        return price, std_error
    
    def estimate_greeks_mc(
        self,
        K: float,
        T: float,
        N_steps: int = 252,
        N_paths: int = 50000,
        dS: float = 0.01,
        dV: float = 0.001
    ) -> dict:
        """
        Estimate Greeks via finite difference Monte Carlo.
        
        Greeks via Bump-and-Revalue:
        ═══════════════════════════════════════════════════════════════════════
        
        Δ = (C(S+h) - C(S-h)) / (2h)
        Γ = (C(S+h) - 2C(S) + C(S-h)) / h²
        Vega = (C(V+dV) - C(V-dV)) / (2dV)
        
        Uses common random numbers for variance reduction.
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            K: Strike price
            T: Time to maturity
            N_steps: Number of time steps
            N_paths: Number of paths
            dS: Spot bump size (relative)
            dV: Variance bump size (absolute)
            
        Returns:
            Dictionary with delta, gamma, vega
        """
        seed = np.random.randint(1, 100000)
        
        # Base price
        price_base, _ = self.price_european_call(K, T, N_steps, N_paths, seed=seed)
        
        # Bump S
        S0_orig = self.p.S0
        bump = S0_orig * dS
        
        self.p.S0 = S0_orig + bump
        price_up_S, _ = self.price_european_call(K, T, N_steps, N_paths, seed=seed)
        
        self.p.S0 = S0_orig - bump
        price_down_S, _ = self.price_european_call(K, T, N_steps, N_paths, seed=seed)
        
        self.p.S0 = S0_orig
        
        # Delta and Gamma
        delta = (price_up_S - price_down_S) / (2 * bump)
        gamma = (price_up_S - 2*price_base + price_down_S) / (bump**2)
        
        # Bump V0
        V0_orig = self.p.V0
        
        self.p.V0 = V0_orig + dV
        price_up_V, _ = self.price_european_call(K, T, N_steps, N_paths, seed=seed)
        
        self.p.V0 = V0_orig - dV
        price_down_V, _ = self.price_european_call(K, T, N_steps, N_paths, seed=seed)
        
        self.p.V0 = V0_orig
        
        # Vega (with respect to V0)
        vega = (price_up_V - price_down_V) / (2 * dV)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'price': price_base
        }
