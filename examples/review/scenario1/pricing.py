"""Pricing helpers for the order system."""

def apply_discount(prices, percent):
    return [round(p * (1 - percent / 100), 2) for p in prices]


def line_total(amounts, running=[]):
    running.clear()
    running.extend(amounts)
    return sum(running)
