"""
test_client.py
--------------
Test script to verify the Semantic Bug Predictor & Auto-Fixer API.

This module uses the `requests` library to send HTTP requests to the
FastAPI backend and verify that both detection and fixing work correctly.

It includes test cases for:
  - Java array out-of-bounds bug
  - Java null pointer dereference
  - C segmentation fault
  - Clean code (no bug)

Usage:
  1. Make sure the API server is running: uvicorn api_server_advanced:app --reload
  2. Run this script: python test_client.py
  3. Check the console output for results

Notes:
  - The script uses `requests` (install via: pip install requests)
  - It's designed to be beginner-friendly with lots of explanatory comments
  - Each test prints the request, response, and interpretation
"""

import requests
import json
from typing import Dict, Any

# ============================================================================
# Configuration
# ============================================================================

# Base URL of the API server
API_BASE_URL = "http://127.0.0.1:8000"

# Color codes for pretty terminal output (optional, for readability)
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# ============================================================================
# Test Cases
# ============================================================================

TEST_CASES = [
    {
        "name": "Java Array Out-of-Bounds",
        "code": """int[] arr = new int[5];
System.out.println(arr[10]);""",
        "expected_bug": True,
    },
    {
        "name": "Java Null Pointer Dereference",
        "code": """String s = null;
System.out.println(s.length());""",
        "expected_bug": True,
    },
    {
        "name": "C Segmentation Fault (Null Pointer)",
        "code": """#include <stdio.h>
int main() {
    int *p = NULL;
    printf("%d", *p);
    return 0;
}""",
        "expected_bug": True,
    },
    {
        "name": "Clean Java Code",
        "code": """public class Main {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}""",
        "expected_bug": False,
    },
]

# ============================================================================
# Helper Functions
# ============================================================================


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")


def print_section(text: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}>>> {text}{Colors.ENDC}")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")


def print_json(data: Dict[str, Any], indent: int = 2):
    """Pretty-print JSON data."""
    print(json.dumps(data, indent=indent))


# ============================================================================
# Test Functions
# ============================================================================


def test_predict(code: str) -> Dict[str, Any]:
    """Test the /predict endpoint.

    Args:
        code: The code snippet to analyze.

    Returns:
        Response JSON dict, or None if request failed.
    """
    print_section("Testing /predict endpoint")
    print(f"Code:\n{Colors.OKBLUE}{code}{Colors.ENDC}\n")

    url = f"{API_BASE_URL}/predict"
    payload = {"code": code}

    print_info(f"POST {url}")
    print_info(f"Payload: {payload}\n")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        print_success("Response received:")
        print_json(data)

        return data
    except requests.exceptions.RequestException as e:
        print_error(f"Request failed: {e}")
        return None


def test_fix(code: str) -> Dict[str, Any]:
    """Test the /fix endpoint.

    Args:
        code: The (presumably buggy) code snippet to fix.

    Returns:
        Response JSON dict, or None if request failed.
    """
    print_section("Testing /fix endpoint")
    print(f"Code:\n{Colors.OKBLUE}{code}{Colors.ENDC}\n")

    url = f"{API_BASE_URL}/fix"
    payload = {"code": code}

    print_info(f"POST {url}")
    print_info(f"Payload: (code snippet)\n")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        print_success("Response received:")
        print_json(data)

        return data
    except requests.exceptions.RequestException as e:
        print_error(f"Request failed: {e}")
        return None


def test_analyze_and_fix(code: str) -> Dict[str, Any]:
    """Test the /analyze_and_fix endpoint (combined detect + fix).

    Args:
        code: The code snippet to analyze and fix.

    Returns:
        Response JSON dict, or None if request failed.
    """
    print_section("Testing /analyze_and_fix endpoint")
    print(f"Code:\n{Colors.OKBLUE}{code}{Colors.ENDC}\n")

    url = f"{API_BASE_URL}/analyze_and_fix"
    payload = {"code": code}

    print_info(f"POST {url}")
    print_info(f"Payload: (code snippet)\n")

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        print_success("Response received:")
        print_json(data)

        return data
    except requests.exceptions.RequestException as e:
        print_error(f"Request failed: {e}")
        return None


def test_health():
    """Test the root endpoint (health check)."""
    print_section("Testing root endpoint (health check)")

    url = API_BASE_URL
    print_info(f"GET {url}\n")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        print_success("API is healthy:")
        print_json(data)

        return True
    except requests.exceptions.RequestException as e:
        print_error(f"Health check failed: {e}")
        print_error("Make sure the API server is running!")
        print_error("Run: uvicorn api_server_advanced:app --reload")
        return False


# ============================================================================
# Main Test Suite
# ============================================================================


def main():
    """Run the complete test suite."""
    print_header("Semantic Bug Predictor & Auto-Fixer Test Suite")

    # Step 1: Health check
    print_info("Step 1: Checking API availability...")
    if not test_health():
        print_error("Cannot connect to API. Exiting.")
        return

    # Step 2: Run test cases
    print_header("Running Test Cases")
    test_count = 0
    passed_count = 0

    for test_case in TEST_CASES:
        test_count += 1
        name = test_case["name"]
        code = test_case["code"]
        expected_bug = test_case["expected_bug"]

        print_header(f"Test {test_count}: {name}")

        # Test predict
        predict_result = test_predict(code)
        if predict_result:
            has_bug = predict_result.get("has_bug")
            confidence = predict_result.get("confidence", 0)

            if has_bug == expected_bug:
                print_success(
                    f"Prediction correct: has_bug={has_bug} (expected={expected_bug})"
                )
                passed_count += 1
            else:
                print_error(
                    f"Prediction incorrect: has_bug={has_bug} (expected={expected_bug})"
                )

            print_info(f"Confidence: {confidence:.4f}")

        # Test fix (only if buggy)
        if predict_result and predict_result.get("has_bug"):
            fix_result = test_fix(code)
            if fix_result:
                fixed_code = fix_result.get("fixed_code")
                if fixed_code:
                    print_success("Fix generated successfully")
                    print_info(f"Fixed code:\n{Colors.OKGREEN}{fixed_code}{Colors.ENDC}")
                else:
                    print_error("Fix generation returned None")

        # Test combined endpoint
        analyze_result = test_analyze_and_fix(code)
        if analyze_result:
            print_success("Combined endpoint works correctly")

    # Final summary
    print_header("Test Summary")
    print(
        f"{Colors.BOLD}Tests Passed: {passed_count}/{test_count}{Colors.ENDC}"
    )

    if passed_count == test_count:
        print_success("All tests passed! 🎉")
    else:
        print_error(f"{test_count - passed_count} test(s) failed")


if __name__ == "__main__":
    main()
