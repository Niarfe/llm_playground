"""Pricing helpers for the order system."""

def apply_discount(prices, percent):
    for i in range(len(prices)):
        prices[i] = round(prices[i] * (1 - percent / 100), 2)
    return prices


def line_total(amounts):
    return sum(amounts)
