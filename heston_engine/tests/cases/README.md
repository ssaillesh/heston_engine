# Validation Cases Audit Guide

This folder contains one validation script per test case for easier auditing.

## How To Run

### Run full suite (recommended)

From workspace root:

```bash
"/Users/saillesh/Desktop/heston 2/.venv/bin/python" heston_engine/tests/validation.py
```

### Run one case directly

From `heston_engine` folder:

```bash
PYTHONPATH=. "/Users/saillesh/Desktop/heston 2/.venv/bin/python" -c "from tests.cases.test_03_method_convergence import test_method_convergence; print(test_method_convergence())"
```

## Case Index

1. `test_01_put_call_parity.py`
   - **Objective:** Verify arbitrage identity `C - P = S·e^{-qT} - K·e^{-rT}`.
   - **Pass criterion:** absolute parity error `< 0.01`.

2. `test_02_bs_limit.py`
   - **Objective:** Verify Heston converges to Black–Scholes when vol-of-vol approaches zero.
   - **Pass criterion:** relative difference `< 1.0%`.

3. `test_03_method_convergence.py`
   - **Objective:** Compare Analytical vs PDE vs Monte Carlo prices.
   - **Pass criteria:**
     - PDE numerically stable
     - PDE fallback not used
     - PDE relative error `< 2.0%`
     - MC relative error `< 2.0%`

4. `test_04_greeks_signs.py`
   - **Objective:** Validate expected sign behavior for call Greeks.
   - **Pass criterion:**
     - `delta > 0`, `gamma > 0`, `vega > 0`, `theta < 0`, `rho > 0`.

5. `test_05_feller_condition.py`
   - **Objective:** Validate Feller condition computation and flags.
   - **Pass criterion:** known satisfying/violating parameter sets are correctly classified.

6. `test_06_mc_variance_positivity.py`
   - **Objective:** Verify QE Monte Carlo keeps variance non-negative.
   - **Pass criterion:** minimum simulated variance `>= 0`.

7. `test_07_moneyness_effects.py`
   - **Objective:** Verify call prices decrease with strike and skew shape is present.
   - **Pass criteria:**
     - call prices strictly decreasing across strikes
     - implied vol skew consistent with negative rho setup.

8. `test_08_grid_convergence.py`
   - **Objective:** Verify PDE error improves on refined grid.
   - **Pass criteria:**
     - coarse/fine PDE outputs are stable
     - fallback not used on either grid
     - fine-grid error `< coarse-grid error`
     - fine-grid absolute error `< 0.1`.

9. `test_09_quantlib_benchmark.py`
   - **Objective:** Benchmark against QuantLib reference prices.
   - **Pass criteria:**
     - analytical MAPE `< 0.10%`
     - analytical max relative error `< 0.50%`
     - MC relative error `< 1.50%`.

## Shared Utilities

- `common.py` contains shared imports, QuantLib helper, and scoring utility.
- All case scripts return `(passed: bool, message: str, details: dict)`.
