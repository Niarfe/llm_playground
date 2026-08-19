"""Run me with: python test_all.py"""

from shapes import rectangle_area, triangle_area
from strings import shout, reverse_words
from numbers import average, clamp


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    print(f"  ok  {label}")


def main():
    print("running tests...")
    check("rectangle_area(3, 4)", rectangle_area(3, 4), 12)
    check("triangle_area(6, 4)", triangle_area(6, 4), 12.0)
    check("shout('hi')", shout("hi"), "HI!")
    check("reverse_words('one two three')",
          reverse_words("one two three"), "three two one")
    check("average([2, 4, 6])", average([2, 4, 6]), 4.0)
    check("clamp(9, 0, 5)", clamp(9, 0, 5), 5)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
