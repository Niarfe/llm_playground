"""
EXTRA — Change how a model reasons by changing the shape of its output.

TAG: quality
    Not on the main line. Chain-of-thought prompting improves answer
    quality; it is not what makes the agent loop function. Remove it and
    08 still works — that is the test for what belongs on the path.

MAIN POINT
    Ask a small model a word problem and it often produces a confident,
    wrong number. Ask it to follow a fixed output structure and accuracy
    frequently improves -- not because it got smarter, but because each
    step now has to be written down where the next step can use it.

THE STRUCTURE USED HERE
    STATE    what is known so far
    GOAL     what this step must find
    MOVE     the operation to perform
    PREDICT  a bound on the answer, committed BEFORE calculating
    OBSERVE  the actual calculation, checked against PREDICT

    PREDICT is the interesting one. "The result is under 48" is cheap to
    produce and cheap to check, but it forces a commitment the model can
    then contradict itself against. OBSERVE catching a mismatch is a
    self-check the model would not otherwise perform.

WHAT ACTUALLY HAPPENS (an observed run, qwen2.5:7b-instruct)
    The structured version got the right answer -- 108 -- but its
    self-checking did not hold up:

      PREDICT: The result is less than 48 + 24 + 36 - 48 = 56
      OBSERVE: 48 + 24 + 36 = 108
      Consistent            <- 108 is not less than 56

    It also wrote PREDICTs like "less than 48 / 2 = 24", which performs the
    calculation inside the prediction -- exactly what the prompt forbids.

    So the structure helped the arithmetic and did NOT deliver a working
    self-check. Both halves of that are the lesson: a model asked to grade
    itself will often write the word "Consistent" because it comes next in
    the pattern, not because it compared anything.

HOW TO USE THIS EXAMPLE
    It runs one question twice -- plain, then structured -- and prints both
    so you can compare. Run it a few times: results vary between runs, and
    noticing that variance is itself part of the lesson. A single side by
    side comparison is an anecdote, not a result.

    Turning anecdotes into results -- fixed question sets, repeated trials,
    scored output -- is a different activity from learning the technique,
    and it needs its own harness rather than a print statement.

RUN IT
    python examples/extras/structured_reasoning.py
"""

import ollama

MODEL = "qwen2.5:7b-instruct"

QUESTION = (
    "A shop sold 48 units in April. In May it sold half as many as April. "
    "In June it sold 12 more than May. How many units did it sell in total "
    "across the three months?"
)

PLAIN_PROMPT = "You are a helpful assistant."

STRUCTURED_PROMPT = """\
Solve the question using this structure.

STATE: restate what is known.
GOAL: restate what must be found.
Then, for each step, one MOVE / PREDICT / OBSERVE cycle:
  MOVE: describe the operation.
  PREDICT: a bound or qualitative claim about the result, made WITHOUT
           calculating. "The result is less than 48 and a whole number."
           A PREDICT that states the exact answer, or that just restates
           the expression ("the answer is 48 / 2"), is invalid.
  OBSERVE: perform the calculation, then say "Consistent" or "Mismatch"
           against the PREDICT. A result equal to a "less than X" bound is
           a Mismatch.
Finish with FINAL ANSWER. Write STATE and GOAL once, at the start.
"""


def ask(system_prompt: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QUESTION},
        ],
        options={"temperature": 0.3},  # low, so runs are roughly comparable
    )
    return response.message.content.strip()


if __name__ == "__main__":
    print(f"QUESTION\n{QUESTION}\n")
    print("Correct answer: 48 + 24 + 36 = 108\n")

    print("=" * 70)
    print("PLAIN PROMPT")
    print("=" * 70)
    print(ask(PLAIN_PROMPT))

    print("\n" + "=" * 70)
    print("STRUCTURED PROMPT")
    print("=" * 70)
    print(ask(STRUCTURED_PROMPT))

    print(
        "\nRun this several times. Note how often each version is right, and "
        "whether OBSERVE ever catches its own mistake."
    )
