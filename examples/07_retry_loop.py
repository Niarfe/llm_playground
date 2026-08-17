"""
07 -- A loop that retries. (This is NOT an agent loop.)

MAIN POINT
    You have just seen tool calling, and the natural next thought is "put it
    in a loop." Do that without care and you build this: a retry loop. It is
    a perfectly good pattern -- it is just not the one that makes an agent.

    A retry loop is REJECTION SAMPLING. Generate, validate, resample on
    failure. It is the standard answer for schema-valid output, for flaky
    networks, and (with a scorer instead of a validator) for best-of-N.

    It works whenever each attempt is an independent draw with a decent
    chance of success. It does NOT make progress: attempt 5 is no wiser
    than attempt 1, because the model is asked exactly the same thing every
    time. Progress comes from variance, not from learning.

THE TEST THAT SEPARATES THE TWO LOOPS
    Does the context change between iterations?

      Retry (here)  same input, resampled output. Progress from variance.
      Agent (08)    input grows with each result. Progress from knowledge.

    The code looks nearly identical -- a while, a budget, a call, a check.
    That similarity is exactly why the mistake is easy to make.

WHEN RETRY FAILS
    When per-attempt success is near zero. Asking an 8B model to solve a
    sudoku and checking the answer will burn 100 attempts and solve nothing:
    independent draws from a distribution that never contains the answer
    never produce the answer. Retrying is not thinking.

THE TEMPERATURE TRAP
    A retry loop at temperature 0 is an infinite loop. Every attempt is
    byte-identical, so if the first fails, all of them fail the same way.
    Retry loops REQUIRE variance to function -- a direct consequence of 03.
    This file demonstrates that deliberately before doing it properly.

PREVIOUSLY
    06 made one tool call. This wraps repetition around a call.

NEXT
    08 is the loop that actually accumulates.

RUN IT
    python examples/07_retry_loop.py
"""

import json

import ollama

MODEL = "llama3.1"
MAX_ATTEMPTS = 6

# Tuning this prompt was an exercise in itself. "Describe a fictional book as
# a JSON object..." fails 10/10 -- llama3.1 opens with "Here is the description"
# and wraps the result in fences. "Give me a JSON object describing..." passes
# 8/10. Same request, same model, same temperature; the framing decides whether
# the output IS JSON or merely CONTAINS JSON.
#
# The year range and exact tag count are here to keep the failure rate honest.
# Without them llama3.1 passes almost every time, and a retry loop that never
# retries demonstrates nothing.
TASK = (
    "Give me a JSON object describing a fictional book, with exactly these "
    "keys: title (string), author (string), year (integer between 1950 and "
    "1990), tags (array of exactly 3 strings). Output ONLY the JSON object."
)


##############################################################################
# THE VALIDATOR                                                  [new in 07]
##############################################################################
# Deliberately strict: the response must parse as JSON on its own. Models love
# to wrap output in ```json fences or open with "Sure!", and both fail here.
# That is realistic -- and it is what gives the loop something to do.

REQUIRED = {"title": str, "author": str, "year": int, "tags": list}


def validate(raw: str):
    """Returns (ok, value_or_reason)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        return False, f"not valid JSON ({error.msg})"

    if not isinstance(data, dict):
        return False, f"expected an object, got {type(data).__name__}"

    for key, expected in REQUIRED.items():
        if key not in data:
            return False, f"missing key '{key}'"
        # bool is a subclass of int in Python; year=True should not pass
        if isinstance(data[key], bool) or not isinstance(data[key], expected):
            return False, f"'{key}' should be {expected.__name__}"

    if not all(isinstance(t, str) for t in data["tags"]):
        return False, "'tags' must contain only strings"

    # Constraints the model can satisfy but often does not -- which is exactly
    # what a validator is for.
    if len(data["tags"]) != 3:
        return False, f"got {len(data['tags'])} tags, need exactly 3"
    if not 1950 <= data["year"] <= 1990:
        return False, f"year {data['year']} outside 1950-1990"

    return True, data


##############################################################################
# THE RETRY LOOP                                                 [new in 07]
##############################################################################
# Note what does NOT happen: the failure reason is never sent back to the
# model. The prompt is identical on every attempt. That is what makes this a
# retry loop rather than an agent loop -- and it is why temperature matters
# so much here.

def retry_loop(temperature: float, max_attempts: int = MAX_ATTEMPTS):
    messages = [{"role": "user", "content": TASK}]

    for attempt in range(1, max_attempts + 1):
        # Same `messages` every time. Nothing accumulates.
        raw = ollama.chat(
            model=MODEL, messages=messages, options={"temperature": temperature}
        ).message.content.strip()

        ok, result = validate(raw)
        preview = " ".join(raw.split())[:70]

        if ok:
            print(f"  attempt {attempt}: OK      | {preview}")
            return result, attempt

        print(f"  attempt {attempt}: REJECTED | {result:34} | {preview}")

    return None, max_attempts


##############################################################################
# THE DEMONSTRATION                                              [new in 07]
##############################################################################

def show_temperature_zero_is_pointless():
    """
    Not "temperature 0 fails" -- it might pass. The point is that it is
    IDENTICAL, so whatever the first attempt does, every later attempt does
    too. A retry loop over a deterministic generator is a for-loop that
    throws away its work.
    """
    print("=" * 76)
    print("TEMPERATURE 0 -- no variance, so retrying cannot change anything")
    print("=" * 76)

    outputs = [
        ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": TASK}],
            options={"temperature": 0},
        ).message.content.strip()
        for _ in range(3)
    ]

    for i, out in enumerate(outputs, start=1):
        ok, _ = validate(out)
        print(f"  attempt {i}: {'OK' if ok else 'REJECTED'} | {' '.join(out.split())[:60]}")

    if len(set(outputs)) == 1:
        print("\n  All three byte-identical. If the first had been rejected, so would")
        print("  every retry, forever. Retry loops REQUIRE variance to function.\n")
    else:
        print("\n  Not identical -- unusual at temperature 0, but sampling is not")
        print("  guaranteed deterministic across all backends.\n")


if __name__ == "__main__":
    show_temperature_zero_is_pointless()

    print("=" * 76)
    print("TEMPERATURE 1.0 -- variance is what gives retrying something to do")
    print("=" * 76)
    result, attempts = retry_loop(temperature=1.0)

    if result:
        print(f"\n  Succeeded on attempt {attempts}:")
        print(f"  {json.dumps(result, indent=2)}")
    else:
        print(f"\n  Gave up after {MAX_ATTEMPTS}. Legitimate outcome -- a budget")
        print("  exists precisely because some loops do not terminate on their own.")

    print(
        "\n  llama3.1 satisfies this schema roughly 8 times in 10, so you may need\n"
        "  a couple of runs to see a rejection. That RATE is the thing that decides\n"
        "  whether rejection sampling is the right tool: at 80% a retry loop is\n"
        "  excellent, at 1% it is a slow way to fail, and at 0% -- the sudoku case --\n"
        "  it never terminates. Measure it before relying on it.\n"
        "\n  Every attempt above asked the exact same question. The model never\n"
        "  learned why it failed. That is the whole difference from 08.\n"
    )
