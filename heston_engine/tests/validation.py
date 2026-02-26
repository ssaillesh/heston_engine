"""
Validation Test Runner

This file aggregates individual validation test cases located under tests/cases.
Each case is in its own script for easier auditing and maintenance.
"""

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.cases.test_01_put_call_parity import test_put_call_parity
from tests.cases.test_02_bs_limit import test_bs_limit
from tests.cases.test_03_method_convergence import test_method_convergence
from tests.cases.test_04_greeks_signs import test_greeks_signs
from tests.cases.test_05_feller_condition import test_feller_condition
from tests.cases.test_06_mc_variance_positivity import test_mc_variance_positivity
from tests.cases.test_07_moneyness_effects import test_moneyness_effects
from tests.cases.test_08_grid_convergence import test_grid_convergence
from tests.cases.test_09_quantlib_benchmark import test_quantlib_benchmark_alignment


class TestResult:
    def __init__(self, name: str, passed: bool, message: str, details: Optional[dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.name} - {self.message}"


def run_test(name: str, test_func) -> TestResult:
    try:
        passed, message, details = test_func()
        return TestResult(name, passed, message, details)
    except Exception as e:
        return TestResult(name, False, f"Exception: {str(e)}")


def run_all_tests() -> bool:
    print("=" * 70)
    print("HESTON ENGINE VALIDATION SUITE")
    print("=" * 70)
    print()

    tests = [
        ("Put-Call Parity", test_put_call_parity),
        ("Black-Scholes Limit", test_bs_limit),
        ("Method Convergence", test_method_convergence),
        ("Greeks Signs", test_greeks_signs),
        ("Feller Condition", test_feller_condition),
        ("MC Variance Positivity", test_mc_variance_positivity),
        ("Moneyness Effects", test_moneyness_effects),
        ("Grid Convergence", test_grid_convergence),
        ("QuantLib Benchmark", test_quantlib_benchmark_alignment),
    ]

    results = []
    for name, test_func in tests:
        result = run_test(name, test_func)
        results.append(result)
        print(result)
        if result.details:
            for key, value in result.details.items():
                if key != 'checks' and not isinstance(value, (list, dict)):
                    print(f"    {key}: {value}")
        print()

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n✓ All validation tests PASSED!")
        return True

    print(f"\n✗ {total - passed} test(s) FAILED")
    return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
