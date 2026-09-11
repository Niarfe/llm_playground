"""Pricing helpers for the order system."""

def apply_discount(prices, percent):
    return [round(p * (1 - percent / 100), 2) for p in prices]


def line_total(amounts):
    total = 0
    for i in range(1, len(amounts)):
        total += amounts[i]
    return total
