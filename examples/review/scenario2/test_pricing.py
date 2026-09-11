"""Run me with: python test_pricing.py"""

from pricing import apply_discount, line_total


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    print(f"  ok  {label}")


def main():
    print("running tests...")
    check("apply_discount([100, 200], 10)", apply_discount([100, 200], 10), [90.0, 180.0])
    check("apply_discount([50], 50)", apply_discount([50], 50), [25.0])
    check("line_total([10, 20, 30])", line_total([10, 20, 30]), 60)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
