"""
Market Data Module for Heston Engine

Free data sources for model parameters:
- Yahoo Finance: Spot prices, options, dividends, historical vol
- FRED: Risk-free rates
- CBOE: VIX data
- US Treasury: Treasury rates
"""

from .fetcher import (
    YahooFinanceFetcher,
    FREDFetcher,
    TreasuryDirectFetcher,
    CBOEFetcher,
    MarketDataAggregator,
    check_dependencies,
    MarketData,
    OptionData,
    RateData,
    YFINANCE_AVAILABLE,
    FRED_AVAILABLE,
    PANDAS_AVAILABLE,
    REQUESTS_AVAILABLE
)

__all__ = [
    'YahooFinanceFetcher',
    'FREDFetcher',
    'TreasuryDirectFetcher',
    'CBOEFetcher',
    'MarketDataAggregator',
    'check_dependencies',
    'MarketData',
    'OptionData',
    'RateData',
    'YFINANCE_AVAILABLE',
    'FRED_AVAILABLE',
    'PANDAS_AVAILABLE',
    'REQUESTS_AVAILABLE'
]
