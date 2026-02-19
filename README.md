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
python run.py --test
python run.py --demo
```
- Put-call parity validation (error monitoring)
- Multi-method consensus testing
- Real-time volatility surface generation
- Live market data calibration pipeline
- Greeks computation (Δ, Γ, Vega via finite differences)
