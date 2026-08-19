"""Small string helpers."""


def shout(text):
    return text.upper() + "!"


def reverse_words(sentence):
    """Reverse the ORDER of words, not the letters."""
    return " ".join(reversed(sentence.split()))
