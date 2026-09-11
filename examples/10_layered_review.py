"""
10 -- Three tiers of defence: lint, tests, reviewer.

WHERE THIS ATTACHES
    09 ends on a limitation it does not solve. Its verifier is a test suite:
    cheap, honest, and completely blind to anything nobody thought to test.
    A bug that passes the tests is, as far as 09 can tell, not a bug.

    So: what catches what the tests cannot?

MAIN POINT
    Run the cheap, certain checks first. Only spend a model on what survives.

        tier 1  LINT      rules. Instant, deterministic, no false confidence.
        tier 2  TESTS     expected outputs. Still deterministic.
        tier 3  REVIEWER  judgement. Slow, fallible, and the only thing left
                          that can notice what nobody encoded.

    The ordering is not politeness, it is economics. Tier 1 costs
    milliseconds and tier 3 costs seconds and can be wrong. Sending a model
    something a linter would have caught is paying the most for the least
    reliable answer.

WHY A REVIEWER IS WORTH ANYTHING AT ALL
    Not because it is smarter. Because it brings INFORMATION THE OTHERS DO
    NOT HAVE.

    A linter knows rules. A test suite knows expected outputs. Both are
    outside knowledge, which is exactly why they are trustworthy. A reviewer
    earns its place only if it contributes something neither holds -- and a
    "critic agent" that is the same weights with a costume prompt contributes
    nothing, because it shares whatever blind spot produced the bug.

    That is the claim worth testing, and this file tests it rather than
    asserting it. Independence of information is the thing. Persona is not.

THE THREE SCENARIOS
    One bug each, pitched at a different tier, so you can watch the ladder
    work rather than take it on faith:

      scenario 1  mutable default argument   -> LINT catches it
      scenario 2  off-by-one in a sum        -> TESTS catch it
      scenario 3  mutates the caller's list  -> both pass. Only tier 3 left.

    Scenario 3 is the one that matters:

        def apply_discount(prices, percent):
            for i in range(len(prices)):
                prices[i] = round(prices[i] * (1 - percent / 100), 2)
            return prices

    Syntactically fine. Lint clean. Returns exactly the right list, so the
    tests pass. And the caller's data is destroyed on the way past.

    The lesson underneath it: TESTS ENCODE WHAT SOMEBODY THOUGHT TO CHECK.
    The defect lives in what nobody thought to check -- which is also why
    "write more tests" is not the answer. You do not know which test is
    missing. That gap is the only honest justification for tier 3.

MEASURED
    Reviewer run against all three scenarios, ignoring the earlier tiers, so
    false positives are visible. Ground truth: does any function here write
    into one of its arguments?

        llama3.1            3/3 correct
        qwen2.5-coder:7b    3/3 correct

    A narrow, evaluative checklist works, and the code-specialised model was
    no better at this particular check. That is a smaller claim than "add a
    critic agent" -- it is one defect class, one prompt, three files.

    Read the first measurement of it, though, which said 2/3 for both models.
    That was wrong, and the oracle was the thing that was wrong: it only
    probed apply_discount, so it scored a TRUE positive on scenario 1 as a
    false one. See bug_is_really_there(). Third time in this repo a narrow
    checker has impersonated a model failure -- suspect your ground truth at
    least as hard as the thing it judges.

RUN IT
    make run-10          # all three scenarios
    python examples/10_layered_review.py --model qwen2.5-coder:7b
"""

import subprocess
import sys
from pathlib import Path

from ollama import chat

MODEL = "llama3.1"
HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "review"

# Narrow and evaluative, per the architecture this example is testing: a
# reviewer that must emit a verdict in a fixed shape drifts far less than one
# invited to muse. Compare 03 -- this is a reporting step, not a creative one.
REVIEWER_PROMPT = """\
You are a code reviewer checking ONE specific class of defect: functions that
modify their arguments.

In Python, lists and dicts are passed by reference. A function that writes into
one of its arguments changes the caller's data. That is a defect when the
function also RETURNS a result, because callers reasonably assume their input
was left alone.

Read the code. Answer in exactly this format and nothing else:

VERDICT: PASS or FAIL
FUNCTION: the function name, or none
REASON: one sentence

FAIL only if a function writes into one of its own arguments. Assigning to a
local name, or building and returning a new list, is fine."""


##############################################################################
# TIER 1 -- LINT                                                 [new in 10]
##############################################################################
# Rules, instantly, with no opinion. Anything this catches should never reach
# a model.

def run_lint(scenario: int) -> tuple[bool, str]:
    result = subprocess.run(
        ["pylint", "--disable=all", "--enable=W0102,W0621,E", "pricing.py"],
        cwd=SCENARIOS / f"scenario{scenario}",
        capture_output=True, text=True, timeout=60,
    )
    findings = [l for l in result.stdout.splitlines() if l.startswith("pricing.py:")]
    return (not findings), (findings[0] if findings else "clean")


##############################################################################
# TIER 2 -- TESTS                                                [new in 10]
##############################################################################
# 09's verifier, unchanged. Expected outputs, honestly compared.

def run_tests(scenario: int) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "test_pricing.py"],
        cwd=SCENARIOS / f"scenario{scenario}",
        capture_output=True, text=True, timeout=30,
    )
    raw = (result.stdout + result.stderr).strip()
    if "ALL TESTS PASSED" in raw:
        return True, "pass"
    errors = [l.strip() for l in raw.splitlines() if "Error:" in l]
    return False, (errors[-1][:60] if errors else "failed")


##############################################################################
# TIER 3 -- REVIEWER                                             [new in 10]
##############################################################################

def run_reviewer(scenario: int, model: str) -> tuple[bool, str]:
    source = (SCENARIOS / f"scenario{scenario}" / "pricing.py").read_text()
    reply = chat(
        model=model,
        messages=[
            {"role": "system", "content": REVIEWER_PROMPT},
            {"role": "user", "content": f"```python\n{source}```"},
        ],
        options={"temperature": 0},
    ).message.content.strip()

    verdict_line = next(
        (l for l in reply.splitlines() if l.upper().startswith("VERDICT")), ""
    )
    passed = "PASS" in verdict_line.upper() and "FAIL" not in verdict_line.upper()
    summary = " ".join(reply.split())[:70]
    return passed, summary


##############################################################################
# THE ORACLE                                                     [new in 10]
##############################################################################
# Ground truth, kept out of every tier. This is how we know whether a tier was
# right -- not by asking it, per 09. It is the test nobody wrote.

def bug_is_really_there(scenario: int) -> bool:
    """
    Does ANY function here write into one of its arguments? Run it and see.

    The first version of this oracle only probed apply_discount, and scored
    the reviewers 2/3 -- both models "wrongly" flagged scenario 1. They were
    right. scenario 1's line_total takes a defaulted `running` list and calls
    .clear() and .extend() on it, which is a mutated argument by any reading,
    and the narrow oracle could not see it.

    So the measurement said the reviewers had a 33% false-positive rate, and
    the reviewers were at 100%. Third time in this repo that a too-narrow
    checker has impersonated a model failure. Whatever you are using as
    ground truth deserves more suspicion than the thing it is judging.
    """
    probe = (
        "import sys; sys.path.insert(0, '.')\n"
        "import pricing\n"
        "before = [100, 200]\n"
        "pricing.apply_discount(before, 10)\n"
        "mutated = before != [100, 200]\n"
        "shared = []\n"
        "try:\n"
        "    pricing.line_total([1, 2], shared)\n"
        "    mutated = mutated or shared != []\n"
        "except TypeError:\n"
        "    pass\n"
        "print('MUTATED' if mutated else 'INTACT')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=SCENARIOS / f"scenario{scenario}",
        capture_output=True, text=True, timeout=30,
    )
    return "MUTATED" in result.stdout


if __name__ == "__main__":
    model = MODEL
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    print(f"[reviewer model] {model}\n")
    print(f"  {'':10}{'TIER 1':<9}{'TIER 2':<9}{'TIER 3':<9}{'MUTATION?':<11}CAUGHT BY")
    print("  " + "-" * 68)

    for scenario in (1, 2, 3):
        lint_ok, lint_note = run_lint(scenario)

        # Only spend later tiers on what survived. This is the economics of
        # the ordering, made literal rather than described.
        if lint_ok:
            tests_ok, test_note = run_tests(scenario)
        else:
            tests_ok, test_note = None, "not run"

        if lint_ok and tests_ok:
            review_ok, review_note = run_reviewer(scenario, model)
        else:
            review_ok, review_note = None, "not run"

        def mark(ok):
            return "clean" if ok else ("CAUGHT" if ok is False else "-")

        caught = ("tier 1" if not lint_ok
                  else "tier 2" if tests_ok is False
                  else "tier 3" if review_ok is False
                  else "NOTHING")

        mutates = "yes" if bug_is_really_there(scenario) else "no"
        print(f"  scenario{scenario} {mark(lint_ok):<9}{mark(tests_ok):<9}"
              f"{mark(review_ok):<9}{mutates:<11}{caught}")

        for note in (lint_note, test_note, review_note):
            if note not in ("clean", "pass", "not run"):
                print(f"  {'':10}{note}")

    print(
        "\n  Scenario 3 is the whole argument. Lint clean, tests green, and the\n"
        "  caller's list destroyed anyway. If tier 3 says NOTHING there, the\n"
        "  reviewer earned nothing -- report that honestly, it is the finding.\n"
        "\n  Then try --model qwen2.5-coder:7b. If a code-specialised model does\n"
        "  better, the useful variable was never 'add a critic', it was 'which\n"
        "  critic, and what does it know that the others do not'.\n"
    )
