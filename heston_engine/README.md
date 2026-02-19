# Heston Stochastic Volatility Model Engine

Numerical pricing and analysis toolkit for the Heston stochastic volatility model, with a Flask API + browser UI for pricing, Greeks, volatility surface visualization, Monte Carlo simulation, validation, and market-data-assisted parameter estimation.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Usage Guide (Web UI)](#usage-guide-web-ui)
8. [API Reference](#api-reference)
9. [Validation & Testing](#validation--testing)
10. [Performance Notes](#performance-notes)
11. [Roadmap Ideas](#roadmap-ideas)
12. [License](#license)

---

## Overview

This project implements the **Heston (1993)** stochastic volatility model for European options and related analytics.

Compared with constant-volatility models (like Black–Scholes), Heston captures:
- Volatility smile/skew
- Stochastic variance dynamics
- Correlation effects between returns and variance

The repository includes:
- Multiple pricing engines (Analytical/Fourier, PDE/ADI, Monte Carlo/QE)
- Greeks computation
- Implied volatility surface generation
- Monte Carlo path visualization
- Validation checks (put-call parity, BS comparison, Feller check)
- Market data endpoints (Yahoo Finance, FRED, CBOE proxies)

---

## Mathematical Formulation

Under risk-neutral dynamics:

$$
dS_t = (r-q)S_t\,dt + \sqrt{V_t}S_t\,dW_t^S
$$

$$
dV_t = \kappa(\theta - V_t)\,dt + \sigma\sqrt{V_t}\,dW_t^V
$$

$$
\mathrm{Corr}(dW_t^S, dW_t^V)=\rho\,dt
$$

Where:
- `kappa` ($\kappa$): mean reversion speed of variance
- `theta` ($\theta$): long-run variance
- `sigma` ($\sigma$): volatility of variance (vol-of-vol)
- `rho` ($\rho$): correlation between price and variance shocks
- `r`: risk-free rate
- `q`: dividend yield
- `S0`: spot price
- `V0`: initial variance

**Feller condition** (strict positivity tendency of CIR variance process):

$$
2\kappa\theta > \sigma^2
$$

---

## Features

- **Pricing Engines**
	- Analytical characteristic-function Fourier inversion
	- PDE solver with ADI scheme
	- Monte Carlo simulator with Andersen QE scheme

- **Risk Analytics**
	- Delta, Gamma, Vega, Theta, Rho
	- Higher-order Greeks: Vanna, Volga

- **Visualization**
	- 3D implied volatility surface
	- Stock and variance Monte Carlo path charts

- **Validation Tools**
	- Put-call parity checks
	- Black–Scholes proximity comparison
	- Feller condition diagnostics

- **Market Data Integration**
	- Spot, dividend, option chain, historical vol from Yahoo Finance
	- Treasury rates via FRED (optional API key)
	- VIX/VVIX proxy via Yahoo ticker endpoints

---

## Tech Stack

### Backend
- Python 3
- Flask + Flask-CORS
- NumPy, SciPy, Pandas
- yfinance, requests
- Optional: fredapi

### Frontend
- HTML/CSS
- Vanilla JavaScript
- Plotly.js (CDN) for charting

### Numerical Methods Used
- Characteristic function + Fourier inversion (analytical Heston pricing)
- Finite differences with ADI splitting (PDE)
- Monte Carlo with Quadratic-Exponential (QE) variance sampling

---

## Project Structure

```text
heston_engine/
├── run.py                        # CLI entrypoint (server, demo, tests)
├── requirements.txt              # Python dependencies
├── README.md
├── __init__.py
│
├── backend/
│   ├── app.py                    # Flask routes + API orchestration
│   ├── __init__.py
│   ├── core/
│   │   ├── parameters.py         # Heston parameter model + defaults
│   │   ├── grid.py               # Numerical grid utilities
│   │   ├── boundaries.py         # Boundary/auxiliary quant functions
│   │   └── __init__.py
│   ├── solvers/
│   │   ├── analytical.py         # Fourier-based pricing engine
│   │   ├── pde_solver.py         # ADI PDE solver
│   │   ├── monte_carlo.py        # QE Monte Carlo simulator
│   │   └── __init__.py
│   ├── greeks/
│   │   ├── calculator.py         # Greeks and higher-order Greeks
│   │   └── __init__.py
│   ├── calibration/
│   │   ├── market_data.py        # Calibration data shaping
│   │   ├── optimizer.py          # Parameter fitting routines
│   │   └── __init__.py
│   ├── data/
│   │   ├── fetcher.py            # Yahoo/FRED/CBOE data adapters
│   │   └── __init__.py
│   └── __pycache__/
│
├── frontend/
│   ├── index.html                # Main pricing/analytics dashboard
│   ├── data.html                 # Market data page
│   ├── app.js                    # Frontend behavior + API calls + Plotly
│   └── style.css
│
└── tests/
		├── validation.py             # Validation suite and consistency tests
		└── __init__.py
```

---

## Quick Start

### 1) Clone and install

```bash
git clone <your-repo-url>
cd heston_engine
pip install -r requirements.txt
```

### 2) Run the application

```bash
PYTHONPATH=.. python3 run.py
```

Default server:
- `http://localhost:5000` (UI)
- `http://localhost:5000/api/health` (health endpoint)

### 3) Optional run modes

```bash
PYTHONPATH=.. python3 run.py --test
PYTHONPATH=.. python3 run.py --demo
PYTHONPATH=.. python3 run.py --port 8000
PYTHONPATH=.. python3 run.py --no-debug
```

---

## Usage Guide (Web UI)

From the main page:

1. Set model parameters (`kappa`, `theta`, `sigma`, `rho`, `r`, `q`, `S0`, `V0`)
2. Set option inputs (`K`, `T`, option type)
3. Choose action:
	 - **Price Option**
	 - **Calculate Greeks**
	 - **Vol Surface**
	 - **MC Paths**
	 - **Compare Methods**
	 - **Validate Model**

The interface also shows a live **Feller condition indicator**:
- Green: likely well-behaved variance positivity regime
- Warning/Fail states: potentially unstable variance behavior under chosen params

---

## API Reference

Base URL: `http://localhost:5000`

### Core endpoints

- `GET /api/health`
- `POST /api/price`
- `POST /api/greeks`
- `POST /api/surface`
- `POST /api/paths`
- `POST /api/compare`
- `POST /api/validate`

### Market data endpoints

- `GET /api/data/check`
- `GET /api/data/spot/<symbol>`
- `GET /api/data/dividend/<symbol>`
- `GET /api/data/volatility/<symbol>?period=1y&window=21`
- `GET /api/data/options/<symbol>?expiration=YYYY-MM-DD`
- `GET /api/data/vix`
- `GET /api/data/rates?api_key=<FRED_KEY>`
- `GET /api/data/all/<symbol>?fred_api_key=<FRED_KEY>`
- `GET /api/data/estimate/<symbol>?fred_api_key=<FRED_KEY>`

### Minimal pricing request example

```bash
curl -X POST http://localhost:5000/api/price \
	-H "Content-Type: application/json" \
	-d '{
		"method": "analytical",
		"params": {
			"kappa": 2.0,
			"theta": 0.04,
			"sigma": 0.3,
			"rho": -0.7,
			"r": 0.05,
			"q": 0.02,
			"S0": 100.0,
			"V0": 0.04
		},
		"K": 100,
		"T": 1.0,
		"option_type": "call"
	}'
```

---

## Validation & Testing

Run suite:

```bash
python run.py --test
```

Validation coverage includes:
- Put-call parity
- Black-Scholes limiting behavior
- Cross-method convergence (Analytical vs PDE vs MC)
- Greeks sign sanity checks
- Monte Carlo statistical consistency checks

---

## Performance Notes

- Analytical pricing is typically the fastest for vanilla options.
- PDE offers deterministic grid-based control at higher computational cost.
- Monte Carlo cost scales with `n_paths × n_steps`.
- For UI responsiveness, very large path counts are expensive to plot in-browser.
- API `POST /api/paths` currently enforces an internal cap on returned path count for visualization safety.

---

## Roadmap Ideas

- Calibration workflow examples and benchmark datasets
- American/exotic option extensions
- Docker packaging for one-command deployment
- CI pipeline with automated numerical regression tests
- Optional auth/rate limits for public API deployment

---

## License

MIT (or your preferred license; update this section if needed).