"""
A keyword baseline. No model, no dependencies, about forty lines.

WHY THIS IS HERE
    "Use an LLM" is a decision, and a decision needs something to be better
    than. Without a floor, any accuracy number sounds good: 70% feels like a
    result right up until you learn that guessing one class gets you 25% and
    a keyword list gets you most of the way.

    So this runs first, every time, and the LLM has to beat it.

A REAL CONTAMINATION, WHICH HAPPENED WRITING THIS FILE
    The first version of this baseline scored 31 out of 32.

    Nothing was tuned. The keyword lists were written once and never
    adjusted against the score. By the usual standard that is clean work.
    It was not clean, and the reason is worth more than the example is.

    The same person had just finished typing the forty tickets, and then
    wrote the keyword lists with those forty tickets still in mind. So the
    lists ended up full of phrases like "does nothing", "do not add up",
    "any chance", "locked out" -- strings lifted out of the test set by
    memory rather than by code. A keyword list that contains the test set's
    phrasing has the answers in it.

    That is the self-confirming evaluation again, with a human in the
    generator's seat instead of a model. No leak you could find by reading
    the code. No train/test boundary crossed. The contamination travelled
    through the author's short-term memory, and it still put eight points of
    fiction on the board.

    The lists below are the rewrite. Every word in them is derivable from the
    four label descriptions in tickets.py and from knowing what a support
    queue looks like. None of them was taken from a ticket.

    The useful habit this suggests: whoever writes the baseline should not be
    the person who just wrote the data, and if it has to be the same person,
    the gap between the two tasks should be long enough to forget.

AND DO NOT TUNE IT NOW
    The tempting move from here is to run it, see which tickets it misses,
    add the words that would have caught them, and run it again. Three
    rounds of that and you have a baseline that scores well on these
    thirty-two tickets and falls over on the thirty-third, because you did
    not build a classifier, you transcribed an answer key by hand. If you
    want to improve it, improve it against tickets you label yourself and
    leave this test set sealed.
"""

from collections import Counter

# Derived from LABEL_NOTES in tickets.py, not from the ticket texts:
#   billing          charges, invoices, refunds, prices, payment methods
#   bug              the product behaves differently from how it should
#   feature_request  something the product does not do yet
#   account_access   logins, passwords, permissions, membership, ownership
KEYWORDS = {
    "billing": [
        "charge", "charged", "invoice", "invoiced", "refund", "price",
        "pricing", "payment", "pay", "paid", "bill", "billed", "billing",
        "receipt", "card", "subscription", "cost", "fee", "currency",
    ],
    "bug": [
        "broken", "break", "bug", "error", "fail", "fails", "failed",
        "crash", "wrong", "incorrect", "glitch", "unexpected", "should not",
        "does not work", "not working", "stopped",
    ],
    "feature_request": [
        "add", "support", "feature", "request", "ability", "option",
        "would like", "could you", "can you", "wish", "suggestion",
        "enhancement", "please let",
    ],
    "account_access": [
        "login", "log in", "sign in", "password", "access", "permission",
        "admin", "owner", "ownership", "member", "membership", "account",
        "two factor", "2fa", "mfa", "sso", "single sign on", "locked",
        "credentials", "invite", "invitation",
    ],
}

# Used when nothing matches. All four classes are equally common here, so
# there is no "most common class" to fall back on and this is an arbitrary
# pick, stated rather than hidden.
FALLBACK = "bug"


def classify(text: str) -> str:
    """Score every class by how many of its keywords appear. Highest wins."""
    lowered = text.lower()
    scores = Counter({
        label: sum(1 for word in words if word in lowered)
        for label, words in KEYWORDS.items()
    })
    best, count = scores.most_common(1)[0]
    return best if count > 0 else FALLBACK
