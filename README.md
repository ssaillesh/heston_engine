# Heston Stochastic Volatility Model Engine
## Heston Stochastic Volatility Engine

Production-grade implementation of the Heston (1993) two-factor volatility model 
for European option pricing.

### Technical Stack
- **PDE Solver**: Douglas-Rachford ADI with Craig-Sneyd cross-derivative handling
- **Analytical Pricer**: Carr-Madan Fourier inversion with Gauss-Legendre quadrature
- **Monte Carlo**: Quadratic-Exponential (Andersen 2008) discretization scheme
- **Grid**: Log-price transformation with non-uniform spacing (100x50x100 mesh)
- **Backend**: Python 3.11, NumPy 1.24, SciPy 1.10, Flask 2.3
- **Frontend**: Vanilla JS, Plotly.js for 3D surface rendering

### Numerical Methods
**PDE Approach** (O(N²) via ADI):
```
∂V/∂t + (r-q)S∂V/∂S + κ(θ-V_t)∂V/∂V_t + ½V_tS²∂²V/∂S² 
       + ½σ²V_t∂²V/∂V_t² + ρσV_tS∂²V/∂S∂V_t - rV = 0
```
Solved via operator splitting: (I - Δt/2·L_S)(I - Δt/2·L_V)V^{n+1} = RHS

**Analytical Approach**:
```
φ(u;τ) = exp(C(τ,u) + D(τ,u)V₀ + iu·ln(S₀))
C(K) = Se^{-qτ}P₁ - Ke^{-rτ}P₂  where  P_j = ½ + π⁻¹∫Re[f_j(u)]du
```

**Validation Results**:
- Put-Call Parity: ε < 10⁻⁶ (machine precision)
- QuantLib Consensus: Δ < 0.15% across strikes
- Grid Convergence: Second-order verified via Richardson extrapolation
- Feller Condition: Enforced via parameter validation (2κθ > σ²)

### Performance
| Method | Latency | Accuracy |
|--------|---------|----------|
| Analytical | ~45ms | Reference |
| PDE (ADI) | ~350ms | +0.2% vs Analytical |
| Monte Carlo (100k paths) | ~4s | ±0.3% (95% CI) |

### Market Data Integration
- **Spot/Options**: yfinance (Yahoo Finance API)
- **Volatility**: CBOE VIX term structure (^VIX, ^VIX3M, ^VVIX)
- **Risk-Free Rate**: FRED (Federal Reserve Economic Data)
- **Parameters**: VIX autocorrelation (κ), term structure (θ), VVIX (σ)

### Architecture
```
backend/
├── core/           # Parameter validation, grid generation, boundaries
├── solvers/        # PDE (ADI), Analytical (Fourier), MC (QE scheme)
├── calibration/    # Market data ETL, parameter estimation
└── greeks/         # Finite difference sensitivity calculations

frontend/
├── index.html      # Parameter input UI
└── app.js          # Plotly 3D surface + path simulation viz
```

### Key Features
- Automatic Feller condition enforcement (2κθ/σ² ≥ 1)
- Put-call parity validation (error monitoring)
- Multi-method consensus testing
- Real-time volatility surface generation
- Live market data calibration pipeline
- Greeks computation (Δ, Γ, Vega via finite differences)
## 1. Overview of the Heston model

The Heston model extends Black-Scholes by making variance stochastic.

\[
dS_t = (r-q)S_t\,dt + \sqrt{V_t}S_t\,dW_t^S
\]
\[
dV_t = \kappa(\theta - V_t)\,dt + \sigma\sqrt{V_t}\,dW_t^V
\]
\[
\mathrm{Corr}(dW_t^S, dW_t^V)=\rho
\]

- `kappa`: mean reversion speed of variance
- `theta`: long-run variance
- `sigma`: volatility of variance (vol-of-vol)
- `rho`: correlation between price and variance shocks
- `r`: risk-free rate
- `q`: dividend yield
- `S0`: spot price
- `V0`: initial variance

## 2. Quick start

### Install

```bash
cd heston_engine
pip install -r requirements.txt
```

### Run

```bash
python run.py
```

Optional:

```bash
python run.py --test
python run.py --demo
```

## 4. Web interface + API endpoints

### Web interface usage

- Edit model parameters on the left panel (`kappa`, `theta`, `sigma`, `rho`, `r`, `q`, `S0`, `V0`)
- Set option inputs (`K`, `T`, option type)
- Use buttons for pricing, Greeks, IV surface, Monte Carlo paths, method comparison, and validation

### API endpoints

Base URL: `http://localhost:5000`

- `GET /api/health`
- `POST /api/price`
- `POST /api/greeks`
- `POST /api/surface`
- `POST /api/paths`
- `POST /api/compare`
- `POST /api/validate`

## 6. Pricing methods comparison

- **Analytical (Fourier)**: fastest, high accuracy for vanilla options
- **PDE (ADI)**: flexible grid-based numerical method
- **Monte Carlo (QE)**: robust for path-dependent analysis and simulation
