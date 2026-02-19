"""
Calibration Engine for Heston Model

═══════════════════════════════════════════════════════════════════════════════
MODEL CALIBRATION VIA OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════════

1. CALIBRATION OBJECTIVE:
   ═══════════════════════════════════════════════════════════════════════════
   
   Find parameters θ* = (κ, θ, σ, ρ, V₀) that best fit market prices:
   
   θ* = argmin Σᵢ wᵢ · [C_market^i - C_model^i(θ)]²
   
   where:
   - C_market^i: Market price of option i
   - C_model^i(θ): Model price of option i with parameters θ
   - wᵢ: Weight for option i

2. WEIGHTING SCHEMES:
   ═══════════════════════════════════════════════════════════════════════════
   
   Equal weights:
   wᵢ = 1
   
   Inverse vega weighting:
   wᵢ = 1/Vega_i
   Gives more weight to options less sensitive to volatility changes
   
   Relative pricing error:
   wᵢ = 1/C_market^i²
   Penalizes relative (percentage) errors equally
   
   Bid-ask spread:
   wᵢ = 1/(Ask_i - Bid_i)²
   More weight to liquid options with tight spreads

3. IMPLIED VOLATILITY SPACE:
   ═══════════════════════════════════════════════════════════════════════════
   
   Alternative: Minimize implied volatility differences
   
   θ* = argmin Σᵢ wᵢ · [σ_IV^market,i - σ_IV^model,i(θ)]²
   
   Advantages:
   - More uniform scale across strikes/maturities
   - Better captures smile shape
   
   Disadvantages:
   - Requires IV inversion (computationally expensive)

4. PARAMETER CONSTRAINTS:
   ═══════════════════════════════════════════════════════════════════════════
   
   Box constraints:
   κ ∈ [0.01, 15]      (mean reversion speed)
   θ ∈ [0.001, 1.0]    (long-term variance)
   σ ∈ [0.01, 3.0]     (vol-of-vol)
   ρ ∈ [-0.99, -0.01]  (correlation, typically negative for equities)
   V₀ ∈ [0.001, 1.0]   (initial variance)
   
   Nonlinear constraint (Feller):
   2κθ > σ²
   
   Can be enforced via penalty or constrained optimization.

5. OPTIMIZATION ALGORITHMS:
   ═══════════════════════════════════════════════════════════════════════════
   
   Global methods (for initial search):
   - Differential Evolution (recommended)
   - Basin Hopping
   - Simulated Annealing
   
   Local methods (for refinement):
   - L-BFGS-B (bounded quasi-Newton)
   - SLSQP (sequential quadratic programming)
   - Nelder-Mead (derivative-free)
   
   Hybrid approach:
   1. Global search with Differential Evolution
   2. Local refinement with L-BFGS-B

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass

from backend.core.parameters import HestonParams, get_calibration_bounds
from backend.solvers.analytical import AnalyticalPricer


@dataclass
class MarketOption:
    """
    Container for market option data.
    """
    strike: float       # Strike price K
    maturity: float     # Time to maturity T (years)
    price: float        # Market price (mid)
    option_type: str    # 'call' or 'put'
    bid: Optional[float] = None  # Bid price
    ask: Optional[float] = None  # Ask price
    volume: Optional[int] = None  # Trading volume
    

class CalibrationEngine:
    """
    Calibration engine for Heston model parameters.
    
    Fits model to market option prices using optimization.
    """
    
    def __init__(
        self, 
        S0: float, 
        r: float, 
        q: float,
        options: List[MarketOption]
    ):
        """
        Initialize calibration engine.
        
        Args:
            S0: Current spot price
            r: Risk-free rate
            q: Dividend yield
            options: List of market options to calibrate to
        """
        self.S0 = S0
        self.r = r
        self.q = q
        self.options = options
        
        # Default bounds
        self.bounds = get_calibration_bounds()
        
        # Calibration results
        self.result = None
        self.calibrated_params = None
    
    def _create_params(self, x: np.ndarray) -> HestonParams:
        """
        Create HestonParams from optimization vector.
        
        Args:
            x: [kappa, theta, sigma, rho, V0]
            
        Returns:
            HestonParams object
        """
        try:
            return HestonParams(
                kappa=x[0],
                theta=x[1],
                sigma=x[2],
                rho=x[3],
                r=self.r,
                q=self.q,
                S0=self.S0,
                V0=x[4]
            )
        except AssertionError:
            # Return default params if parameters invalid
            return HestonParams(
                kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
                r=self.r, q=self.q, S0=self.S0, V0=0.04
            )
    
    def _compute_weights(
        self, 
        weighting: str = 'equal'
    ) -> np.ndarray:
        """
        Compute calibration weights.
        
        Weighting Schemes:
        ═══════════════════════════════════════════════════════════════════════
        
        'equal': wᵢ = 1
        
        'vega': wᵢ = 1/Vega_i (inverse Black-Scholes vega)
        
        'relative': wᵢ = 1/C_market^i (inverse price)
        
        'spread': wᵢ = 1/(Ask - Bid)² (inverse spread squared)
        
        ═══════════════════════════════════════════════════════════════════════
        """
        n = len(self.options)
        
        if weighting == 'equal':
            return np.ones(n)
        
        elif weighting == 'relative':
            return 1.0 / np.maximum(np.array([o.price for o in self.options]), 0.01)
        
        elif weighting == 'spread':
            weights = np.ones(n)
            for i, opt in enumerate(self.options):
                if opt.bid is not None and opt.ask is not None:
                    spread = max(opt.ask - opt.bid, 0.01)
                    weights[i] = 1.0 / spread**2
            return weights
        
        else:
            return np.ones(n)
    
    def _objective(
        self, 
        x: np.ndarray, 
        weights: np.ndarray
    ) -> float:
        """
        Calibration objective function.
        
        Objective:
        ═══════════════════════════════════════════════════════════════════════
        
        f(θ) = Σᵢ wᵢ · [C_market^i - C_model^i(θ)]²
        
        With Feller penalty:
        f(θ) += λ · max(0, σ² - 2κθ)²
        
        ═══════════════════════════════════════════════════════════════════════
        """
        params = self._create_params(x)
        if params is None:
            return 1e10  # Large penalty for invalid params
        
        pricer = AnalyticalPricer(params)
        
        total_error = 0.0
        
        for i, opt in enumerate(self.options):
            try:
                if opt.option_type == 'call':
                    model_price = pricer.call_price(opt.strike, opt.maturity)
                else:
                    model_price = pricer.put_price(opt.strike, opt.maturity)
                
                error = (opt.price - model_price)**2
                total_error += weights[i] * error
                
            except:
                total_error += 1e6  # Penalty for computation failure
        
        # ═══════════════════════════════════════════════════════════════════
        # FELLER CONDITION PENALTY
        # ═══════════════════════════════════════════════════════════════════
        #
        # Soft constraint: penalize violation of 2κθ > σ²
        #
        # Penalty = λ · max(0, σ² - 2κθ)²
        # ═══════════════════════════════════════════════════════════════════
        
        feller_violation = max(0, x[2]**2 - 2*x[0]*x[1])
        feller_penalty = 1000 * feller_violation**2
        
        return total_error + feller_penalty
    
    def calibrate(
        self,
        method: str = 'differential_evolution',
        weighting: str = 'equal',
        maxiter: int = 1000,
        verbose: bool = True
    ) -> HestonParams:
        """
        Calibrate Heston parameters to market data.
        
        Calibration Algorithm:
        ═══════════════════════════════════════════════════════════════════════
        
        1. Differential Evolution (global search):
           - Population-based stochastic optimizer
           - Explores parameter space efficiently
           - Avoids local minima
        
        2. L-BFGS-B refinement (local search):
           - Gradient-based quasi-Newton method
           - Fast local convergence
           - Respects box constraints
        
        ═══════════════════════════════════════════════════════════════════════
        
        Args:
            method: 'differential_evolution', 'lbfgsb', or 'hybrid'
            weighting: Weight scheme ('equal', 'relative', 'spread')
            maxiter: Maximum iterations
            verbose: Print progress
            
        Returns:
            Calibrated HestonParams
        """
        weights = self._compute_weights(weighting)
        
        # Bounds as list of tuples
        bounds_list = [
            self.bounds['kappa'],
            self.bounds['theta'],
            self.bounds['sigma'],
            self.bounds['rho'],
            self.bounds['V0']
        ]
        
        if verbose:
            print("Starting calibration...")
            print(f"  Method: {method}")
            print(f"  Options: {len(self.options)}")
            print(f"  Weighting: {weighting}")
        
        if method == 'differential_evolution':
            # ═══════════════════════════════════════════════════════════════
            # DIFFERENTIAL EVOLUTION
            # ═══════════════════════════════════════════════════════════════
            #
            # Stochastic population-based optimizer:
            # 1. Initialize random population
            # 2. For each generation:
            #    a. Mutation: combine random individuals
            #    b. Crossover: mix with target
            #    c. Selection: keep if better
            # 3. Converge to global optimum
            # ═══════════════════════════════════════════════════════════════
            
            result = differential_evolution(
                lambda x: self._objective(x, weights),
                bounds_list,
                maxiter=maxiter,
                tol=1e-7,
                strategy='best1bin',
                disp=verbose,
                workers=-1,  # Use all CPU cores
                updating='deferred'
            )
            
        elif method == 'lbfgsb':
            # Initial guess (middle of bounds)
            x0 = np.array([
                (b[0] + b[1]) / 2 for b in bounds_list
            ])
            
            result = minimize(
                lambda x: self._objective(x, weights),
                x0,
                method='L-BFGS-B',
                bounds=bounds_list,
                options={'maxiter': maxiter, 'disp': verbose}
            )
            
        elif method == 'hybrid':
            # Step 1: Global search with DE
            if verbose:
                print("  Step 1: Global search (DE)")
            
            result_de = differential_evolution(
                lambda x: self._objective(x, weights),
                bounds_list,
                maxiter=maxiter // 2,
                tol=1e-5,
                disp=False
            )
            
            # Step 2: Local refinement with L-BFGS-B
            if verbose:
                print("  Step 2: Local refinement (L-BFGS-B)")
            
            result = minimize(
                lambda x: self._objective(x, weights),
                result_de.x,
                method='L-BFGS-B',
                bounds=bounds_list,
                options={'maxiter': maxiter // 2, 'disp': False}
            )
        else:
            # Default to differential evolution if method not recognized
            result = differential_evolution(
                lambda x: self._objective(x, weights),
                bounds_list,
                maxiter=maxiter,
                tol=1e-7
            )
        
        self.result = result
        self.calibrated_params = self._create_params(result.x)
        
        if verbose:
            print(f"\nCalibration complete!")
            print(f"  Final objective: {result.fun:.6f}")
            print(f"  Calibrated parameters:")
            print(f"    κ = {result.x[0]:.4f}")
            print(f"    θ = {result.x[1]:.4f}")
            print(f"    σ = {result.x[2]:.4f}")
            print(f"    ρ = {result.x[3]:.4f}")
            print(f"    V₀ = {result.x[4]:.4f}")
            
            if self.calibrated_params:
                print(f"  Feller ratio: {self.calibrated_params.feller_ratio:.3f}")
        
        return self.calibrated_params
    
    def compute_fit_metrics(self) -> Dict:
        """
        Compute calibration fit metrics.
        
        Returns dictionary with:
        - RMSE: Root mean squared error
        - MAE: Mean absolute error
        - MAPE: Mean absolute percentage error
        - Individual errors per option
        """
        if self.calibrated_params is None:
            raise ValueError("Must calibrate first")
        
        pricer = AnalyticalPricer(self.calibrated_params)
        
        errors = []
        for opt in self.options:
            if opt.option_type == 'call':
                model_price = pricer.call_price(opt.strike, opt.maturity)
            else:
                model_price = pricer.put_price(opt.strike, opt.maturity)
            
            errors.append({
                'strike': opt.strike,
                'maturity': opt.maturity,
                'market_price': opt.price,
                'model_price': model_price,
                'error': model_price - opt.price,
                'pct_error': (model_price - opt.price) / opt.price * 100
            })
        
        errors_arr = np.array([e['error'] for e in errors])
        pct_errors = np.array([e['pct_error'] for e in errors])
        
        return {
            'rmse': np.sqrt(np.mean(errors_arr**2)),
            'mae': np.mean(np.abs(errors_arr)),
            'mape': np.mean(np.abs(pct_errors)),
            'max_error': np.max(np.abs(errors_arr)),
            'individual_errors': errors
        }


def create_synthetic_market_data(
    params: HestonParams,
    strikes: np.ndarray,
    maturities: np.ndarray,
    noise_std: float = 0.0
) -> List[MarketOption]:
    """
    Create synthetic market data for testing calibration.
    
    Args:
        params: True Heston parameters
        strikes: Array of strike prices
        maturities: Array of maturities
        noise_std: Standard deviation of price noise (0 = exact)
        
    Returns:
        List of MarketOption objects
    """
    pricer = AnalyticalPricer(params)
    options = []
    
    for T in maturities:
        for K in strikes:
            price = pricer.call_price(K, T)
            
            # Add noise if specified
            if noise_std > 0:
                price += np.random.normal(0, noise_std * price)
                price = max(price, 0.01)  # Ensure positive
            
            options.append(MarketOption(
                strike=K,
                maturity=T,
                price=price,
                option_type='call'
            ))
    
    return options
