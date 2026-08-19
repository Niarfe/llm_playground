"""Small numeric helpers."""


def average(values):
    """Mean of a list of numbers."""
    return sum(values) / len(values)


def clamp(value, low, high):
    return max(low, min(value, high))
