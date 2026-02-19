"""
Flask Backend API for Heston Engine

═══════════════════════════════════════════════════════════════════════════════
REST API ENDPOINTS FOR HESTON MODEL OPERATIONS
═══════════════════════════════════════════════════════════════════════════════

Endpoints:
- POST /api/price: Calculate option price
- POST /api/greeks: Calculate Greeks
- POST /api/surface: Generate implied volatility surface
- POST /api/paths: Simulate Monte Carlo paths
- POST /api/calibrate: Calibrate model to market data
- GET /api/health: Health check

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.parameters import HestonParams, get_default_params
from backend.core.grid import Grid, create_default_grid
from backend.core.boundaries import BoundaryConditions
from backend.solvers.pde_solver import PDESolver
from backend.solvers.analytical import AnalyticalPricer
from backend.solvers.monte_carlo import MonteCarloSimulator
from backend.greeks.calculator import GreeksCalculator


# Initialize Flask app
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_json_data() -> dict:
    """Get JSON data from request with fallback to empty dict."""
    json_data = request.json
    return json_data if json_data is not None else {}


def parse_params(data: dict) -> HestonParams:
    """
    Parse HestonParams from request data.
    
    Expected format:
    {
        "params": {
            "kappa": float,
            "theta": float,
            "sigma": float,
            "rho": float,
            "r": float,
            "q": float,
            "S0": float,
            "V0": float
        }
    }
    """
    p = data.get('params', {})
    return HestonParams(
        kappa=float(p.get('kappa', 2.0)),
        theta=float(p.get('theta', 0.04)),
        sigma=float(p.get('sigma', 0.3)),
        rho=float(p.get('rho', -0.7)),
        r=float(p.get('r', 0.05)),
        q=float(p.get('q', 0.02)),
        S0=float(p.get('S0', 100.0)),
        V0=float(p.get('V0', 0.04))
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def serve_frontend():
    """Serve the frontend HTML."""
    return send_from_directory(STATIC_FOLDER, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory(STATIC_FOLDER, path)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Heston Engine API',
        'version': '1.0.0'
    })


@app.route('/api/price', methods=['POST'])
def price_option():
    """
    Calculate option price using selected method.
    
    Request JSON:
    {
        "method": "analytical" | "pde" | "monte_carlo",
        "params": {kappa, theta, sigma, rho, r, q, S0, V0},
        "K": strike,
        "T": maturity,
        "option_type": "call" | "put",
        "mc_paths": number of MC paths (optional, default 100000),
        "mc_steps": number of MC time steps (optional, default 252)
    }
    
    Response JSON:
    {
        "price": float,
        "method": string,
        "stderr": float (for MC only),
        "feller_ratio": float,
        "feller_satisfied": bool
    }
    """
    try:
        data = get_json_data()
        
        # Parse parameters
        params = parse_params(data)
        K = float(data.get('K', params.S0))
        T = float(data.get('T', 1.0))
        method = data.get('method', 'analytical')
        option_type = data.get('option_type', 'call')
        
        result = {
            'feller_ratio': params.feller_ratio,
            'feller_satisfied': params.feller_satisfied
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # ANALYTICAL PRICING (Fourier Inversion)
        # ═══════════════════════════════════════════════════════════════════
        
        if method == 'analytical':
            pricer = AnalyticalPricer(params)
            if option_type == 'call':
                price = pricer.call_price(K, T)
            else:
                price = pricer.put_price(K, T)
            
            result.update({
                'price': float(price),
                'method': 'Analytical (Fourier Inversion)'
            })
        
        # ═══════════════════════════════════════════════════════════════════
        # PDE SOLVER (ADI Method)
        # ═══════════════════════════════════════════════════════════════════
        
        elif method == 'pde':
            grid = create_default_grid(params, T, K)
            solver = PDESolver(params, grid)
            
            if option_type == 'call':
                U_grid = solver.solve_european_call(K)
            else:
                U_grid = solver.solve_european_put(K)
            
            price = solver.get_price(U_grid, params.S0, params.V0)
            
            result.update({
                'price': float(price),
                'method': 'PDE Solver (ADI)',
                'grid_size': f"{grid.N_S}x{grid.N_V}x{grid.N_t}"
            })
        
        # ═══════════════════════════════════════════════════════════════════
        # MONTE CARLO (QE Scheme)
        # ═══════════════════════════════════════════════════════════════════
        
        elif method == 'monte_carlo':
            mc = MonteCarloSimulator(params)
            n_paths = int(data.get('mc_paths', 100000))
            n_steps = int(data.get('mc_steps', 252))
            
            if option_type == 'call':
                price, stderr = mc.price_european_call(K, T, n_steps, n_paths)
            else:
                price, stderr = mc.price_european_put(K, T, n_steps, n_paths)
            
            result.update({
                'price': float(price),
                'stderr': float(stderr),
                'method': f'Monte Carlo (QE, {n_paths:,} paths)',
                'confidence_95': [float(price - 1.96*stderr), float(price + 1.96*stderr)]
            })
        
        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/greeks', methods=['POST'])
def compute_greeks():
    """
    Calculate option Greeks.
    
    Request JSON:
    {
        "params": {kappa, theta, sigma, rho, r, q, S0, V0},
        "K": strike,
        "T": maturity,
        "option_type": "call" | "put"
    }
    
    Response JSON:
    {
        "price": float,
        "delta": float,
        "gamma": float,
        "vega": float,
        "theta": float,
        "rho": float,
        "vanna": float,
        "volga": float
    }
    """
    try:
        data = get_json_data()
        params = parse_params(data)
        K = float(data.get('K', params.S0))
        T = float(data.get('T', 1.0))
        option_type = data.get('option_type', 'call')
        
        calculator = GreeksCalculator(params)
        greeks = calculator.all_greeks(K, T, option_type)
        
        # Convert to serializable format
        result = {k: float(v) for k, v in greeks.items()}
        result['theta_daily'] = result['theta'] / 365  # Daily theta
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/surface', methods=['POST'])
def implied_vol_surface():
    """
    Generate implied volatility surface.
    
    Request JSON:
    {
        "params": {kappa, theta, sigma, rho, r, q, S0, V0},
        "K_min": float (default: 0.7*S0),
        "K_max": float (default: 1.3*S0),
        "n_strikes": int (default: 15),
        "T_min": float (default: 0.1),
        "T_max": float (default: 2.0),
        "n_maturities": int (default: 10)
    }
    
    Response JSON:
    {
        "strikes": [floats],
        "maturities": [floats],
        "surface": [[floats]],  # [maturity][strike]
        "prices": [[floats]]    # [maturity][strike]
    }
    """
    try:
        data = get_json_data()
        params = parse_params(data)
        
        S0 = params.S0
        K_min = float(data.get('K_min', S0 * 0.7))
        K_max = float(data.get('K_max', S0 * 1.3))
        n_strikes = int(data.get('n_strikes', 15))
        T_min = float(data.get('T_min', 0.1))
        T_max = float(data.get('T_max', 2.0))
        n_maturities = int(data.get('n_maturities', 10))
        
        strikes = np.linspace(K_min, K_max, n_strikes)
        maturities = np.linspace(T_min, T_max, n_maturities)
        
        pricer = AnalyticalPricer(params)
        
        # Compute price surface
        prices = pricer.price_surface(strikes, maturities, 'call')
        
        # Compute implied volatility surface
        iv_surface = pricer.implied_vol_surface(strikes, maturities)
        
        return jsonify({
            'strikes': strikes.tolist(),
            'maturities': maturities.tolist(),
            'surface': np.nan_to_num(iv_surface, nan=0.2).tolist(),
            'prices': prices.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/paths', methods=['POST'])
def simulate_paths():
    """
    Simulate Monte Carlo paths for visualization.
    
    Request JSON:
    {
        "params": {kappa, theta, sigma, rho, r, q, S0, V0},
        "T": float (default: 1.0),
        "n_steps": int (default: 252),
        "n_paths": int (default: 10)
    }
    
    Response JSON:
    {
        "times": [floats],
        "S_paths": [[floats]],  # [path][time]
        "V_paths": [[floats]]   # [path][time]
    }
    """
    try:
        data = get_json_data()
        params = parse_params(data)
        
        T = float(data.get('T', 1.0))
        n_steps = int(data.get('n_steps', 252))
        n_paths = min(int(data.get('n_paths', 10)), 50)  # Limit for performance
        
        mc = MonteCarloSimulator(params)
        S_paths, V_paths = mc.simulate_paths_qe(T, n_steps, n_paths)
        
        times = np.linspace(0, T, n_steps + 1)
        
        return jsonify({
            'times': times.tolist(),
            'S_paths': S_paths.tolist(),
            'V_paths': V_paths.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def compare_methods():
    """
    Compare prices from all three methods.
    
    Request JSON:
    {
        "params": {kappa, theta, sigma, rho, r, q, S0, V0},
        "K": strike,
        "T": maturity
    }
    
    Response JSON:
    {
        "analytical": {"price": float, "time_ms": float},
        "pde": {"price": float, "time_ms": float},
        "monte_carlo": {"price": float, "stderr": float, "time_ms": float}
    }
    """
    import time
    
    try:
        data = get_json_data()
        params = parse_params(data)
        K = float(data.get('K', params.S0))
        T = float(data.get('T', 1.0))
        
        result = {}
        
        # Analytical
        t0 = time.time()
        pricer = AnalyticalPricer(params)
        price_analytical = pricer.call_price(K, T)
        result['analytical'] = {
            'price': float(price_analytical),
            'time_ms': (time.time() - t0) * 1000
        }
        
        # PDE
        t0 = time.time()
        grid = create_default_grid(params, T, K)
        solver = PDESolver(params, grid)
        U_grid = solver.solve_european_call(K)
        price_pde = solver.get_price(U_grid, params.S0, params.V0)
        result['pde'] = {
            'price': float(price_pde),
            'time_ms': (time.time() - t0) * 1000
        }
        
        # Monte Carlo
        t0 = time.time()
        mc = MonteCarloSimulator(params)
        price_mc, stderr = mc.price_european_call(K, T, 252, 50000)
        result['monte_carlo'] = {
            'price': float(price_mc),
            'stderr': float(stderr),
            'time_ms': (time.time() - t0) * 1000
        }
        
        # Compute differences
        result['differences'] = {
            'pde_vs_analytical': float(price_pde - price_analytical),
            'mc_vs_analytical': float(price_mc - price_analytical),
            'pde_vs_analytical_pct': float((price_pde - price_analytical) / price_analytical * 100),
            'mc_vs_analytical_pct': float((price_mc - price_analytical) / price_analytical * 100)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/validate', methods=['POST'])
def validate_model():
    """
    Run validation tests on the model.
    
    Tests:
    1. Put-call parity
    2. Black-Scholes limit (σ → 0)
    3. Convergence tests
    
    Response JSON:
    {
        "put_call_parity": {"passed": bool, "error": float},
        "bs_limit": {"passed": bool, "heston_price": float, "bs_price": float},
        "convergence": {"passed": bool, "max_error": float}
    }
    """
    try:
        data = get_json_data()
        params = parse_params(data)
        K = float(data.get('K', params.S0))
        T = float(data.get('T', 1.0))
        
        pricer = AnalyticalPricer(params)
        results = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # PUT-CALL PARITY TEST
        # C - P = S·e^{-qT} - K·e^{-rT}
        # ═══════════════════════════════════════════════════════════════════
        
        call_price = pricer.call_price(K, T)
        put_price = pricer.put_price(K, T)
        parity_theoretical = params.S0 * np.exp(-params.q * T) - K * np.exp(-params.r * T)
        parity_actual = call_price - put_price
        parity_error = abs(parity_actual - parity_theoretical)
        
        results['put_call_parity'] = {
            'passed': bool(parity_error < 0.01),
            'call_price': float(call_price),
            'put_price': float(put_price),
            'theoretical': float(parity_theoretical),
            'actual': float(parity_actual),
            'error': float(parity_error)
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # BLACK-SCHOLES LIMIT TEST (σ → 0, ρ = 0)
        # ═══════════════════════════════════════════════════════════════════
        
        from backend.core.boundaries import BoundaryConditions
        bc = BoundaryConditions()
        
        # BS price with σ = √θ (long-term vol)
        bs_sigma = np.sqrt(params.theta)
        bs_price = bc.black_scholes_call(params.S0, K, T, params.r, params.q, bs_sigma)
        
        # Heston price should be close when V0 = θ and rho = 0
        results['bs_comparison'] = {
            'heston_price': float(call_price),
            'bs_price': float(bs_price),
            'difference': float(call_price - bs_price),
            'difference_pct': float((call_price - bs_price) / bs_price * 100)
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # FELLER CONDITION CHECK
        # ═══════════════════════════════════════════════════════════════════
        
        results['feller_condition'] = {
            'ratio': float(params.feller_ratio),
            'satisfied': bool(params.feller_satisfied),
            'two_kappa_theta': float(2 * params.kappa * params.theta),
            'sigma_squared': float(params.sigma ** 2)
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # OVERALL STATUS
        # ═══════════════════════════════════════════════════════════════════
        
        results['overall'] = {
            'passed': bool(results['put_call_parity']['passed']),
            'message': 'All validation tests passed' if results['put_call_parity']['passed'] 
                      else 'Some validation tests failed'
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# DATA SOURCES API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/data')
def serve_data_page():
    """Serve the data sources HTML page."""
    return send_from_directory(STATIC_FOLDER, 'data.html')


@app.route('/api/data/check', methods=['GET'])
def check_data_sources():
    """Check which data source dependencies are available."""
    try:
        from backend.data.fetcher import check_dependencies
        deps = check_dependencies()
        return jsonify({
            'success': True,
            'dependencies': deps,
            'install_commands': {
                'yfinance': 'pip install yfinance',
                'fredapi': 'pip install fredapi',
                'pandas': 'pip install pandas',
                'requests': 'pip install requests'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/spot/<symbol>', methods=['GET'])
def get_spot_price(symbol):
    """Fetch spot price for a symbol."""
    try:
        from backend.data.fetcher import YahooFinanceFetcher
        fetcher = YahooFinanceFetcher()
        result = fetcher.get_spot_price(symbol.upper())
        return jsonify(result)
    except ImportError as e:
        return jsonify({'success': False, 'error': 'yfinance not installed', 'install': 'pip install yfinance'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/dividend/<symbol>', methods=['GET'])
def get_dividend(symbol):
    """Fetch dividend yield for a symbol."""
    try:
        from backend.data.fetcher import YahooFinanceFetcher
        fetcher = YahooFinanceFetcher()
        result = fetcher.get_dividend_yield(symbol.upper())
        return jsonify(result)
    except ImportError as e:
        return jsonify({'success': False, 'error': 'yfinance not installed', 'install': 'pip install yfinance'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/volatility/<symbol>', methods=['GET'])
def get_historical_vol(symbol):
    """Fetch historical volatility for a symbol."""
    try:
        from backend.data.fetcher import YahooFinanceFetcher
        fetcher = YahooFinanceFetcher()
        period = request.args.get('period', '1y')
        window = int(request.args.get('window', 21))
        result = fetcher.get_historical_volatility(symbol.upper(), period, window)
        return jsonify(result)
    except ImportError as e:
        return jsonify({'success': False, 'error': 'yfinance not installed', 'install': 'pip install yfinance'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/options/<symbol>', methods=['GET'])
def get_options(symbol):
    """Fetch option chain for a symbol."""
    try:
        from backend.data.fetcher import YahooFinanceFetcher
        fetcher = YahooFinanceFetcher()
        expiration = request.args.get('expiration')
        result = fetcher.get_option_chain(symbol.upper(), expiration)
        return jsonify(result)
    except ImportError as e:
        return jsonify({'success': False, 'error': 'yfinance not installed', 'install': 'pip install yfinance'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/vix', methods=['GET'])
def get_vix():
    """Fetch VIX data."""
    try:
        from backend.data.fetcher import CBOEFetcher
        fetcher = CBOEFetcher()
        result = fetcher.get_vix_data()
        return jsonify(result)
    except ImportError as e:
        return jsonify({'success': False, 'error': 'yfinance not installed', 'install': 'pip install yfinance'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/rates', methods=['GET'])
def get_rates():
    """Fetch treasury rates from FRED."""
    try:
        from backend.data.fetcher import FREDFetcher
        api_key = request.args.get('api_key')
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'FRED API key required',
                'note': 'Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html',
                'fallback_rates': {
                    '3m': 0.05,
                    '1y': 0.05,
                    'note': 'Using fallback rate of 5%'
                }
            })
        fetcher = FREDFetcher(api_key)
        result = fetcher.get_treasury_rates()
        return jsonify(result)
    except ImportError as e:
        return jsonify({'success': False, 'error': 'fredapi not installed', 'install': 'pip install fredapi'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/all/<symbol>', methods=['GET'])
def get_all_data(symbol):
    """Fetch all available market data for a symbol."""
    try:
        from backend.data.fetcher import MarketDataAggregator
        fred_api_key = request.args.get('fred_api_key')
        aggregator = MarketDataAggregator(fred_api_key)
        result = aggregator.get_all_data(symbol.upper())
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/estimate/<symbol>', methods=['GET'])
def estimate_params(symbol):
    """Estimate initial Heston parameters from market data."""
    try:
        from backend.data.fetcher import MarketDataAggregator
        fred_api_key = request.args.get('fred_api_key')
        aggregator = MarketDataAggregator(fred_api_key)
        result = aggregator.get_heston_params_estimate(symbol.upper())
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("HESTON STOCHASTIC VOLATILITY MODEL ENGINE")
    print("=" * 70)
    print()
    print("Mathematical Model:")
    print("  dS = (r-q)S dt + √V S dW_S")
    print("  dV = κ(θ-V)dt + σ√V dW_V")
    print("  Corr(dW_S, dW_V) = ρ")
    print()
    print("Starting Flask server...")
    print("  API: http://localhost:5000/api")
    print("  UI:  http://localhost:5000")
    print()
    print("=" * 70)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
