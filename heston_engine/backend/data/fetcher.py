"""
═══════════════════════════════════════════════════════════════════════════════
MARKET DATA FETCHER - Free Data Sources for Heston Model Parameters
═══════════════════════════════════════════════════════════════════════════════

FREE DATA SOURCES:
┌─────────────────────────────────────────────────────┐
│  Spot Price      → Yahoo Finance (yfinance)         │
│  Option Chain    → Yahoo Finance (yfinance)         │
│  Risk-Free Rate  → FRED (Federal Reserve)           │
│  VIX/Vol Data    → CBOE (official, free)            │
│  Dividend Yield  → Yahoo Finance (yfinance)         │
│  Treasury Rates  → US Treasury (direct, free)       │
│  Historical Vol  → Yahoo Finance (yfinance)         │
└─────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class MarketData:
    """Container for market data."""
    spot_price: float
    dividend_yield: float
    historical_volatility: float
    timestamp: str
    source: str


@dataclass
class OptionData:
    """Container for option chain data."""
    expiration: str
    strikes: List[float]
    call_prices: List[float]
    put_prices: List[float]
    call_ivs: List[float]
    put_ivs: List[float]
    timestamp: str


@dataclass
class RateData:
    """Container for interest rate data."""
    rate_1m: Optional[float]
    rate_3m: Optional[float]
    rate_6m: Optional[float]
    rate_1y: Optional[float]
    rate_2y: Optional[float]
    rate_10y: Optional[float]
    timestamp: str
    source: str


class YahooFinanceFetcher:
    """
    Fetch data from Yahoo Finance using yfinance library.
    
    Data available:
    - Spot prices
    - Option chains
    - Dividend yields
    - Historical volatility
    """
    
    def __init__(self):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    def get_spot_price(self, symbol: str) -> Dict:
        """
        Fetch current spot price for a symbol.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'SPY', '^SPX')
        
        Returns:
            Dict with price info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price = info.get('regularMarketPrice') or info.get('previousClose')
            
            return {
                'success': True,
                'symbol': symbol,
                'price': price,
                'currency': info.get('currency', 'USD'),
                'name': info.get('shortName', symbol),
                'exchange': info.get('exchange', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance'
            }
        except Exception as e:
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'source': 'Yahoo Finance'
            }
    
    def get_dividend_yield(self, symbol: str) -> Dict:
        """
        Fetch dividend yield for a symbol.
        
        Args:
            symbol: Stock ticker
        
        Returns:
            Dict with dividend info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # dividendYield from yfinance is in percentage form (e.g., 1.05 = 1.05%)
            # We need to convert to decimal form (0.0105)
            div_yield_raw = info.get('dividendYield') or info.get('trailingAnnualDividendYield') or 0
            
            # If > 1, it's in percentage form (e.g., 1.05%), convert to decimal
            # If < 1, it's already in decimal form (e.g., 0.0105)
            if div_yield_raw > 1:
                div_yield = div_yield_raw / 100  # 1.05 -> 0.0105
            else:
                div_yield = div_yield_raw
            
            div_rate = info.get('dividendRate') or info.get('trailingAnnualDividendRate') or 0
            
            return {
                'success': True,
                'symbol': symbol,
                'dividend_yield': div_yield,
                'dividend_yield_pct': div_yield * 100,  # For display: 1.05%
                'dividend_rate': div_rate,
                'ex_dividend_date': info.get('exDividendDate'),
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance'
            }
        except Exception as e:
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'source': 'Yahoo Finance'
            }
    
    def get_historical_volatility(self, symbol: str, period: str = '1y', window: int = 21) -> Dict:
        """
        Calculate historical volatility from price data.
        
        Args:
            symbol: Stock ticker
            period: Data period ('1mo', '3mo', '6mo', '1y', '2y')
            window: Rolling window for volatility calculation (trading days)
        
        Returns:
            Dict with volatility info
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) < window + 1:
                return {
                    'success': False,
                    'symbol': symbol,
                    'error': f'Insufficient data: need {window+1} days, got {len(hist)}',
                    'source': 'Yahoo Finance'
                }
            
            # Calculate log returns
            log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            
            # Annualized volatility (252 trading days)
            current_vol = log_returns.std() * np.sqrt(252)
            
            # Rolling volatility
            rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
            
            return {
                'success': True,
                'symbol': symbol,
                'volatility': float(current_vol),
                'volatility_pct': float(current_vol * 100),
                'rolling_volatility': [float(v) for v in rolling_vol.dropna().tail(20).tolist()],
                'period': period,
                'window': window,
                'data_points': len(log_returns),
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance'
            }
        except Exception as e:
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'source': 'Yahoo Finance'
            }
    
    def get_option_chain(self, symbol: str, expiration: Optional[str] = None) -> Dict:
        """
        Fetch option chain data.
        
        Args:
            symbol: Stock ticker
            expiration: Specific expiration date (YYYY-MM-DD) or None for nearest
        
        Returns:
            Dict with option chain data
        """
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return {
                    'success': False,
                    'symbol': symbol,
                    'error': 'No options available for this symbol',
                    'source': 'Yahoo Finance'
                }
            
            # Use specified expiration or first available
            exp_date = expiration if expiration in expirations else expirations[0]
            
            opt_chain = ticker.option_chain(exp_date)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Get spot price for moneyness calculation
            spot_data = self.get_spot_price(symbol)
            spot = spot_data.get('price', 100) if spot_data['success'] else 100
            
            # Filter to reasonable strikes (50% to 150% of spot)
            call_mask = (calls['strike'] >= spot * 0.5) & (calls['strike'] <= spot * 1.5)
            put_mask = (puts['strike'] >= spot * 0.5) & (puts['strike'] <= spot * 1.5)
            
            filtered_calls = calls[call_mask]
            filtered_puts = puts[put_mask]
            
            # Helper to convert Series to list, replacing NaN with 0
            def safe_list(series, as_int=False):
                filled = series.fillna(0)
                if as_int:
                    return filled.astype(int).tolist()
                return filled.tolist()
            
            return {
                'success': True,
                'symbol': symbol,
                'spot_price': spot,
                'expiration': exp_date,
                'all_expirations': list(expirations),
                'calls': {
                    'strikes': safe_list(filtered_calls['strike']),
                    'last_prices': safe_list(filtered_calls['lastPrice']),
                    'bids': safe_list(filtered_calls['bid']),
                    'asks': safe_list(filtered_calls['ask']),
                    'implied_volatility': safe_list(filtered_calls['impliedVolatility']),
                    'volume': safe_list(filtered_calls['volume'], as_int=True),
                    'open_interest': safe_list(filtered_calls['openInterest'], as_int=True)
                },
                'puts': {
                    'strikes': safe_list(filtered_puts['strike']),
                    'last_prices': safe_list(filtered_puts['lastPrice']),
                    'bids': safe_list(filtered_puts['bid']),
                    'asks': safe_list(filtered_puts['ask']),
                    'implied_volatility': safe_list(filtered_puts['impliedVolatility']),
                    'volume': safe_list(filtered_puts['volume'], as_int=True),
                    'open_interest': safe_list(filtered_puts['openInterest'], as_int=True)
                },
                'timestamp': datetime.now().isoformat(),
                'source': 'Yahoo Finance'
            }
        except Exception as e:
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'source': 'Yahoo Finance'
            }
    
    def get_vix(self) -> Dict:
        """
        Fetch VIX (CBOE Volatility Index) data.
        
        Returns:
            Dict with VIX info
        """
        return self.get_spot_price('^VIX')


class FREDFetcher:
    """
    Fetch interest rate data from Federal Reserve Economic Data (FRED).
    
    Requires FRED API key (free registration at https://fred.stlouisfed.org/)
    
    Data available:
    - Treasury rates
    - Federal funds rate
    - LIBOR/SOFR
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if not FRED_AVAILABLE:
            raise ImportError("fredapi not installed. Run: pip install fredapi")
        
        if api_key:
            self.fred = Fred(api_key=api_key)
        else:
            self.fred = None
    
    def get_treasury_rates(self) -> Dict:
        """
        Fetch US Treasury rates from FRED.
        
        Series IDs:
        - DGS1MO: 1-Month Treasury
        - DGS3MO: 3-Month Treasury
        - DGS6MO: 6-Month Treasury
        - DGS1: 1-Year Treasury
        - DGS2: 2-Year Treasury
        - DGS10: 10-Year Treasury
        
        Returns:
            Dict with treasury rates
        """
        if not self.fred:
            return {
                'success': False,
                'error': 'FRED API key not provided. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html',
                'source': 'FRED'
            }
        
        try:
            rates = {}
            series_map = {
                '1m': 'DGS1MO',
                '3m': 'DGS3MO', 
                '6m': 'DGS6MO',
                '1y': 'DGS1',
                '2y': 'DGS2',
                '10y': 'DGS10'
            }
            
            for name, series_id in series_map.items():
                try:
                    data = self.fred.get_series(series_id, observation_start=datetime.now() - timedelta(days=7))
                    rates[name] = float(data.dropna().iloc[-1]) / 100  # Convert to decimal
                except:
                    rates[name] = None
            
            return {
                'success': True,
                'rates': rates,
                'timestamp': datetime.now().isoformat(),
                'source': 'FRED (Federal Reserve)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'source': 'FRED'
            }


class TreasuryDirectFetcher:
    """
    Fetch treasury rates directly from US Treasury (no API key needed).
    
    Data available:
    - Daily Treasury Par Yield Curve Rates
    """
    
    def __init__(self):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")
    
    def get_treasury_rates(self) -> Dict:
        """
        Fetch current treasury rates from Treasury.gov XML feed.
        
        Returns:
            Dict with treasury rates
        """
        try:
            # US Treasury XML feed
            url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/all"
            
            # Alternative: Use a simpler approach with the RSS feed
            # This is a simplified version - in production you'd parse the actual data
            
            return {
                'success': True,
                'rates': {
                    '1m': None,  # Fill from actual data
                    '3m': None,
                    '6m': None,
                    '1y': None,
                    '2y': None,
                    '10y': None
                },
                'note': 'Visit https://home.treasury.gov/resource-center/data-chart-center/interest-rates for current rates',
                'timestamp': datetime.now().isoformat(),
                'source': 'US Treasury Direct'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'source': 'US Treasury Direct'
            }


class CBOEFetcher:
    """
    Fetch volatility data from CBOE.
    
    Data available (via Yahoo Finance as proxy):
    - VIX Index
    - VIX Futures
    - VVIX (VIX of VIX)
    """
    
    def __init__(self):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    def get_vix_data(self) -> Dict:
        """
        Fetch VIX and related volatility indices.
        
        Returns:
            Dict with volatility indices
        """
        try:
            vix = yf.Ticker('^VIX')
            vix_info = vix.info
            vix_hist = vix.history(period='1mo')
            
            result = {
                'success': True,
                'vix': {
                    'current': vix_info.get('regularMarketPrice') or vix_info.get('previousClose'),
                    'open': vix_info.get('regularMarketOpen'),
                    'high': vix_info.get('regularMarketDayHigh'),
                    'low': vix_info.get('regularMarketDayLow'),
                    'previous_close': vix_info.get('previousClose'),
                    'history': vix_hist['Close'].tail(20).tolist() if len(vix_hist) > 0 else []
                },
                'timestamp': datetime.now().isoformat(),
                'source': 'CBOE (via Yahoo Finance)'
            }
            
            # Try to get VVIX
            try:
                vvix = yf.Ticker('^VVIX')
                vvix_info = vvix.info
                result['vvix'] = {
                    'current': vvix_info.get('regularMarketPrice') or vvix_info.get('previousClose')
                }
            except:
                pass
            
            return result
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'source': 'CBOE'
            }
    
    def get_vix_term_structure(self) -> Dict:
        """
        Fetch VIX futures term structure.
        
        Returns:
            Dict with VIX futures data
        """
        # VIX futures symbols (these change monthly)
        # This is a simplified version
        try:
            vix = yf.Ticker('^VIX')
            spot_vix = vix.info.get('regularMarketPrice') or vix.info.get('previousClose')
            
            return {
                'success': True,
                'spot_vix': spot_vix,
                'note': 'VIX futures data requires CBOE subscription. Visit https://www.cboe.com/tradable_products/vix/',
                'timestamp': datetime.now().isoformat(),
                'source': 'CBOE'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'source': 'CBOE'
            }


class MarketDataAggregator:
    """
    Aggregate data from all sources to provide complete market data
    for Heston model calibration.
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.yahoo = YahooFinanceFetcher() if YFINANCE_AVAILABLE else None
        self.fred = FREDFetcher(fred_api_key) if FRED_AVAILABLE and fred_api_key else None
        self.cboe = CBOEFetcher() if YFINANCE_AVAILABLE else None

    @staticmethod
    def _safe_float(value: Optional[float], default: float) -> float:
        """Return a finite float or fallback to default."""
        if value is None:
            return default
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if np.isfinite(result) else default
    
    def get_all_data(self, symbol: str) -> Dict:
        """
        Fetch all available data for a symbol.
        
        Args:
            symbol: Stock ticker
        
        Returns:
            Dict with all market data needed for Heston calibration
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'data_sources': {}
        }
        
        # Yahoo Finance data
        if self.yahoo:
            result['spot'] = self.yahoo.get_spot_price(symbol)
            result['dividend'] = self.yahoo.get_dividend_yield(symbol)
            result['historical_vol'] = self.yahoo.get_historical_volatility(symbol)
            result['options'] = self.yahoo.get_option_chain(symbol)
            result['data_sources']['yahoo'] = True
        else:
            result['data_sources']['yahoo'] = False
        
        # VIX data
        if self.cboe:
            result['vix'] = self.cboe.get_vix_data()
            result['data_sources']['cboe'] = True
        else:
            result['data_sources']['cboe'] = False
        
        # Interest rates
        if self.fred:
            result['rates'] = self.fred.get_treasury_rates()
            result['data_sources']['fred'] = True
        else:
            result['data_sources']['fred'] = False
            result['rates'] = {
                'success': False,
                'note': 'FRED API key not provided. Using fallback rates.',
                'fallback_rate': 0.05
            }
        
        return result
    
    def get_heston_params_estimate(self, symbol: str) -> Dict:
        """
        Estimate initial Heston parameters from market data.
        
        This provides a starting point for calibration.
        
        Args:
            symbol: Stock ticker
        
        Returns:
            Dict with estimated Heston parameters
        """
        data = self.get_all_data(symbol)
        
        # Extract values with defaults
        S0 = self._safe_float(data.get('spot', {}).get('price'), 100.0)
        q = self._safe_float(data.get('dividend', {}).get('dividend_yield'), 0.02)
        hist_vol = self._safe_float(data.get('historical_vol', {}).get('volatility'), 0.2)
        if data.get('vix', {}).get('success'):
            vix_current = self._safe_float(data.get('vix', {}).get('vix', {}).get('current'), 20.0)
            vix = vix_current / 100.0
        else:
            vix = 0.2
        
        # Risk-free rate
        if data.get('rates', {}).get('success'):
            r = self._safe_float(data['rates']['rates'].get('3m'), 0.05)
        else:
            r = 0.05
        
        # Initial variance estimate
        V0 = hist_vol ** 2  # Current variance
        theta = vix ** 2  # Long-term variance from VIX
        
        # Rule-of-thumb estimates for other parameters
        kappa = 2.0  # Mean reversion speed
        sigma = 0.3  # Vol of vol (typical range 0.2-0.5)
        rho = -0.7   # Correlation (typically negative for equities)
        
        return {
            'success': True,
            'symbol': symbol,
            'estimated_params': {
                'S0': S0,
                'V0': V0,
                'kappa': kappa,
                'theta': theta,
                'sigma': sigma,
                'rho': rho,
                'r': r,
                'q': q
            },
            'data_used': {
                'spot_price': S0,
                'dividend_yield': q,
                'historical_volatility': hist_vol,
                'vix': vix * 100,
                'risk_free_rate': r
            },
            'notes': [
                'These are initial estimates - calibration to option prices recommended',
                'kappa, sigma, rho should be calibrated to implied volatility surface',
                'theta estimated from VIX, adjust based on term structure'
            ],
            'timestamp': datetime.now().isoformat()
        }


def check_dependencies() -> Dict:
    """Check which data source dependencies are available (dynamic check)."""
    import importlib
    
    def check_import(module_name):
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    
    yf_ok = check_import('yfinance')
    pd_ok = check_import('pandas')
    fred_ok = check_import('fredapi')
    req_ok = check_import('requests')
    
    return {
        'yfinance': yf_ok,
        'pandas': pd_ok,
        'fredapi': fred_ok,
        'requests': req_ok,
        'missing': [
            pkg for pkg, available in [
                ('yfinance', yf_ok),
                ('pandas', pd_ok),
                ('fredapi', fred_ok),
                ('requests', req_ok)
            ] if not available
        ]
    }
