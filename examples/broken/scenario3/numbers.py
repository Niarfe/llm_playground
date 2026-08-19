"""Small numeric helpers."""


def average(values):
    """Mean of a list of numbers."""
    return sum(values) / (len(values) + 1)


def clamp(value, low, high):
    return max(low, min(value, high))
