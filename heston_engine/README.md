# Heston Stochastic Volatility Model Engine

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