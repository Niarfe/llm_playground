"""
07a -- Constrained output, or: how to not need 07's loop.

STATUS: PARTIALLY VERIFIED
    The mechanism is confirmed -- schema-constrained calls return parseable
    JSON, checked directly against Ollama. The SUCCESS RATES below are NOT
    measured yet. Run it and fill them in; do not quote them until you have.

    That distinction is the repo's whole standard. See notes.md.

WHERE THIS ATTACHES
    This is a deepening of 07, not a step past it. 07 teaches rejection
    sampling: ask, validate, resample on failure. Everything it says is
    true and the technique is standard.

    But 07's specific loop exists for one reason -- the model was free to
    wrap its JSON in prose, or in markdown fences, or to ignore the schema.
    Take that freedom away and the loop has nothing left to do.

MAIN POINT
    Before reaching for a retry loop, ask whether the failure it retries
    can be made impossible instead.

    Constrained decoding does exactly that. Ollama accepts a JSON schema in
    `format=`, and the sampler is then restricted at each step to tokens
    that keep the output valid against it. Not "asked nicely" -- the invalid
    tokens are not available to pick.

    This is the same lesson as gating in 08, one layer down: constrain the
    interface rather than instruct the model. There it removed a tool from
    the menu; here it removes a token from the distribution.

WHAT IT DOES NOT FIX
    Structure is not correctness. The schema guarantees you get an integer
    in `year`. It does not guarantee the year is right, or that the book
    exists. Validation of MEANING still belongs to you -- the part of 07's
    validator that checks 1950 <= year <= 1990 still has a job.

    So the honest split:
      schema      -> shape. Free, total, no retries.
      your code   -> meaning. Still yours. Still needs a loop sometimes.

RUN IT
    python examples/07a_structured_output.py

    It runs the same request three ways and counts outcomes. Fill in the
    STATUS block from what you see.
"""

import json

import ollama

MODEL = "llama3.1"
TRIALS = 5

TASK = (
    "Give me a JSON object describing a fictional book, with exactly these "
    "keys: title (string), author (string), year (integer between 1950 and "
    "1990), tags (array of exactly 3 strings)."
)

# The schema is the constraint. Note it can express shape -- types, required
# keys, array length -- but not the 1950-1990 rule, which is a fact about
# values. That gap is the point of the last section in the header.
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"type": "string"},
        "year": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"},
                 "minItems": 3, "maxItems": 3},
    },
    "required": ["title", "author", "year", "tags"],
}


##############################################################################
# THE THREE WAYS TO ASK                                         [new in 07a]
##############################################################################

def ask_unconstrained(prompt: str) -> str:
    """How 07 does it: ask politely, hope for the best."""
    return ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt + " Output ONLY the JSON object."}],
        options={"temperature": 1.0},
    ).message.content.strip()


def ask_json_mode(prompt: str) -> str:
    """format="json" -- valid JSON guaranteed, but any shape at all."""
    return ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 1.0},
    ).message.content.strip()


def ask_schema(prompt: str) -> str:
    """format=<schema> -- valid JSON matching YOUR shape."""
    return ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=SCHEMA,
        options={"temperature": 1.0},
    ).message.content.strip()


##############################################################################
# CHECKING                                                      [new in 07a]
##############################################################################
# Two separate questions, deliberately reported apart, because constrained
# decoding answers the first and cannot answer the second.

def check_shape(raw: str):
    """Is it parseable JSON with the right keys and types?"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, "not JSON"

    for key, kind in {"title": str, "author": str, "year": int, "tags": list}.items():
        if key not in data:
            return False, f"missing {key}"
        if isinstance(data[key], bool) or not isinstance(data[key], kind):
            return False, f"{key} wrong type"
    if len(data["tags"]) != 3:
        return False, f"{len(data['tags'])} tags, need 3"
    return True, data


def check_meaning(data):
    """Is the CONTENT within the stated bounds? A schema cannot know this."""
    if not 1950 <= data["year"] <= 1990:
        return False, f"year {data['year']} outside 1950-1990"
    return True, "ok"


if __name__ == "__main__":
    print(f"[model] {MODEL}   [trials] {TRIALS} per method\n")

    for label, ask in [
        ("unconstrained (07's way)", ask_unconstrained),
        ('format="json"', ask_json_mode),
        ("format=SCHEMA", ask_schema),
    ]:
        shape_ok = meaning_ok = 0
        first_failure = ""

        for _ in range(TRIALS):
            raw = ask(TASK)
            ok, result = check_shape(raw)
            if ok:
                shape_ok += 1
                good, why = check_meaning(result)
                meaning_ok += good
                if not good and not first_failure:
                    first_failure = why
            elif not first_failure:
                first_failure = f"{result}: {' '.join(raw.split())[:50]}"

        print(f"{label:26} shape {shape_ok}/{TRIALS}   meaning {meaning_ok}/{TRIALS}")
        if first_failure:
            print(f"{'':26} first failure: {first_failure}")

    print(
        "\nRead the two columns separately. Constraining the format should move\n"
        "the SHAPE column and leave the MEANING column roughly alone -- a schema\n"
        "cannot know that 2019 is outside the range you asked for.\n"
        "\nWhere shape is 100%, 07's retry loop has nothing to retry. Where\n"
        "meaning still fails, you still need one.\n"
    )
