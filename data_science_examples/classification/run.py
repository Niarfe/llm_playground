"""
CLASSIFICATION -- three ways, and the number that is not accuracy.

WHERE THIS ATTACHES
    text_2_sql graded itself against a database, which has no opinion. This
    task does not get that. The grader here is a column of labels a person
    typed, and people disagree. So this is the first example where the
    measurement itself has to be examined, not just read.

MAIN POINT
    Four hundred labelled tickets and a logistic regression was the right
    answer to this problem for twenty years, and it still often is. An LLM
    changes the economics -- no training set, no training run, a new class is
    a line of prose -- and changes nothing about whether it works.

    So run all three and look:

        RULES      a hand-written keyword list. No model. Milliseconds.
        ZERO-SHOT  the four labels, described, and nothing else.
        FEW-SHOT   the same, plus two labelled examples per class.

    The rules baseline is not a joke entry. It exists so that the LLM numbers
    have a floor to clear. A model that scores 70% where keywords score 60%
    bought you ten points for a GPU and a second per ticket, and that is a
    trade somebody should make on purpose rather than by default.

THE LEAK, WHICH IS ONE LINE OF CODE AWAY
    Eight of the forty tickets are marked train. Those eight are the only
    ones few-shot may show the model. The other thirty-two are the test set
    and the model never sees them labelled.

    Get that wrong -- sample the examples from the whole set, which is the
    natural thing to write -- and the model is sometimes shown the answer to
    the question you are about to ask it. The score goes up. Nothing has
    improved. This is the single most common way a promising evaluation turns
    out to be worthless, and it does not announce itself: the output looks
    like a good result, because a good result is exactly what it looks like.

THE CEILING IS NOT 100%
    Five test tickets are marked ambiguous, because they have a real claim on
    two labels:

        #35  "I cannot see the billing section even though I am an owner."

    Permissions, or a bug in how permissions are displayed? Both. The label
    says account_access because somebody had to choose, and a model that says
    bug is not obviously wrong.

    So every accuracy number below is reported twice, once over all thirty-two
    and once over the twenty-seven that are not ambiguous. The gap between
    those two numbers is the part of the score that is really measuring
    agreement with one annotator's judgement calls.

    Nearly every published benchmark has this gap. Nearly none of them
    report it.

THE NUMBER THAT IS NOT ACCURACY
    Each LLM strategy is also asked how confident it is, and the run prints
    stated confidence against observed accuracy at that confidence.

    A calibrated classifier that says 0.7 is right about 70% of the time. If
    it says 0.95 on everything, including the ones it gets wrong, then the
    confidence is decoration -- it tells you nothing you did not already
    know, and worse, it tells you something false with great assurance.

    This matters past the curiosity. If confidence tracked reality you could
    route the low-confidence tickets to a person and automate the rest, which
    is the only version of this system worth deploying. Calibration is what
    makes an unreliable classifier useful. Its absence is what makes a
    reliable-sounding one dangerous.

    Note also what the model CANNOT do: return a probability over all four
    labels. It emits one label and a number it wrote about itself. Models
    built for this task -- a fine-tuned classifier with a softmax, or a
    decision model like TypeSafe's Jev -- return a distribution, which is a
    different and more useful object.

OUTPUT DISCIPLINE
    A classifier whose output space is "any string" is not a classifier.
    Unparseable replies are counted and reported separately rather than
    quietly mapped to a label, because mapping them hides the failure, and a
    hidden failure in the parsing layer looks exactly like a model that
    works.

MEASURED
    llama3.1, temperature 0, 2026-11-10, M1 Mac.

        rules       24/32 = 75%     20/27 = 74% excluding the ambiguous five
        zero-shot   28/32 = 88%     23/27 = 85%     (1 reply unparseable)
        few-shot    31/32 = 97%     27/27 = 100%

    Five things in that table, in order of how much they change what you
    would do next.

      1. THE LLM EARNED ITS KEEP, WHICH WAS NOT GUARANTEED. Thirteen points
         over a keyword list, and another nine from eight examples. Worth
         saying plainly because the baseline was put there to make the
         opposite outcome possible, and on plenty of tasks it happens.

         Where the keyword list actually broke is informative: 4 of 8 on
         feature_request, against 6-7 of 8 everywhere else. The words that
         define a feature request -- "add", "support", "option" -- are the
         most generic words in a support queue, so "An API endpoint for bulk
         user creation would save us hours" scored as a bug. Keywords work
         on classes with distinctive vocabulary and fall apart on classes
         defined by intent. That is not a fixable bug in the list. It is the
         shape of the method.

      2. FEW-SHOT IS AT THE CEILING, NOT NEAR IT. 27 of 27 on the
         unambiguous tickets. Its single miss was #19, "Scheduled emails
         went out twice this morning", which is marked ambiguous because a
         schedule configured twice is not a bug. The model said billing,
         which is harder to defend, but the point stands: there is no
         remaining headroom on this set that is not argument about labels.

         Which means this test set is now too small and too easy to measure
         further progress. Knowing that is the result. Chasing 98% here
         would be measuring noise.

      3. FEW-SHOT FIXED THE FORMAT, SEPARATELY FROM FIXING THE ANSWERS.
         Zero-shot produced one reply that could not be parsed at all, on
         #5, "Please cancel my plan and refund the last payment". Few-shot
         produced none. Demonstrations teach the output shape as much as the
         task, and output shape is what breaks pipelines at 3am. Worth
         counting on its own rather than folding into accuracy.

      4. ZERO-SHOT WAS BETTER CALIBRATED THAN EXPECTED.

              said 0.70-0.90    right 5/7  = 71%
              said 0.90-1.00    right 23/24 = 96%

         Those track. Not perfectly, and on seven and twenty-four samples
         nothing is established, but the direction is right and the
         low-confidence bucket really did contain more of the errors. A
         route-the-uncertain-to-a-human design has something to work with
         here. This file was written expecting to show the opposite, and the
         expectation was wrong.

      5. AND FEW-SHOT BROKE THE CALIBRATION -- BECAUSE OF A BUG IN THIS FILE.

              said 0.70-0.90    right 1/2  = 50%
              said 0.90-1.00    right 30/30 = 100%

         Thirty of thirty-two replies came back at 0.9 or above. The
         confidence signal went flat, and the reason is in few_shot_messages()
         below: all eight demonstrations are hard-coded as `label|0.95`.

         So the examples did not only teach the model which label to pick.
         They taught it what number to put after the pipe. It copied the
         number, as it should -- that is what a demonstration is -- and the
         accuracy went up while the confidence stopped carrying information.

         Two defects wearing one coat. The honest fix is to vary the
         confidence across the demonstrations so there is nothing to anchor
         on, which is a one-line change and is deliberately NOT made yet,
         because the measurement above is worth keeping as the before.

         The general lesson is larger than the bug: every incidental detail
         of a few-shot example is being taught, not just the part you meant
         to teach. Formatting, length, hedging, tone, and here a constant. If
         all your examples share an accident, the model learns the accident.

    WHAT THIS DOES NOT SHOW
         One model, one run, thirty-two tickets, four classes, one annotator.
         Confidence intervals on 32 samples are wide enough to swallow most
         of the gaps above. The ladder from 75 to 88 to 97 is probably real
         because it is large; the difference between 88% and 91% on a set
         this size would not be.

RUN
    make run-classify
    python data_science_examples/classification/run.py --model qwen2.5:7b-instruct
"""

import argparse
import re
import sys
from collections import Counter, defaultdict

import ollama

import baseline
import tickets as ticket_set

DEFAULT_MODEL = "llama3.1"

LABEL_BLOCK = "\n".join(
    f"  {name:<16} {note}" for name, note in ticket_set.LABEL_NOTES.items()
)

SYSTEM_PROMPT = f"""\
You label customer support tickets. The four labels are:

{LABEL_BLOCK}

Reply with exactly one line, in this form:

  label|confidence

where label is one of {', '.join(ticket_set.LABELS)} and confidence is a
number between 0 and 1 saying how likely it is that your label is correct.
No other text.
"""

REPLY = re.compile(r"\b(billing|bug|feature_request|account_access)\b\s*\|?\s*"
                   r"([01](?:\.\d+)?)?", re.IGNORECASE)


def parse(reply: str) -> tuple[str | None, float | None]:
    """
    Pull a label and a confidence out of the reply.

    Deliberately tolerant about whitespace, case and a missing confidence,
    and deliberately strict about the label: if no known label appears, this
    returns None and the ticket is counted as unparseable rather than being
    assigned to something.
    """
    match = REPLY.search(reply.strip())
    if not match:
        return None, None
    label = match.group(1).lower()
    confidence = float(match.group(2)) if match.group(2) else None
    return label, confidence


def few_shot_messages() -> list[dict]:
    """
    The eight train tickets, as a worked conversation. Test set excluded.

    KNOWN DEFECT, LEFT IN ON PURPOSE -- see point 5 under MEASURED.

    Every demonstration below says 0.95. The model copies that, so after
    few-shot prompting it reports 0.9+ on nearly everything and the
    confidence signal goes flat. Few-shot bought nine points of accuracy and
    spent the calibration to get them.

    The fix is to vary these numbers. It has not been applied because the
    measured run above is the evidence for the problem, and the problem is
    more useful to a student than the fix is. Vary them and re-run to watch
    the calibration come back.
    """
    messages = []
    for ticket in ticket_set.TRAIN:
        messages.append({"role": "user", "content": ticket["text"]})
        messages.append({"role": "assistant", "content": f"{ticket['label']}|0.95"})
    return messages


def ask(model: str, text: str, shots: list[dict]) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                 + shots
                 + [{"role": "user", "content": text}],
        options={"temperature": 0, "num_ctx": 4096},
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

class Result:
    def __init__(self, name: str):
        self.name = name
        self.rows: list[tuple[dict, str | None, float | None]] = []

    def add(self, ticket, predicted, confidence):
        self.rows.append((ticket, predicted, confidence))

    def accuracy(self, skip_ambiguous: bool = False) -> tuple[int, int]:
        hits = total = 0
        for ticket, predicted, _ in self.rows:
            if skip_ambiguous and ticket.get("ambiguous"):
                continue
            total += 1
            hits += predicted == ticket["label"]
        return hits, total

    @property
    def unparseable(self) -> int:
        return sum(1 for _, predicted, _ in self.rows if predicted is None)

    def per_class(self) -> dict[str, tuple[int, int]]:
        out = {}
        for label in ticket_set.LABELS:
            subset = [r for r in self.rows if r[0]["label"] == label]
            out[label] = (sum(1 for t, p, _ in subset if p == label), len(subset))
        return out

    def confusion(self) -> dict[tuple[str, str], int]:
        return Counter((t["label"], p or "unparseable") for t, p, _ in self.rows)

    def calibration(self) -> list[tuple[str, int, int]]:
        """(bucket, correct, total) for replies that gave a confidence."""
        buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        out = []
        for low, high in buckets:
            subset = [(t, p) for t, p, c in self.rows
                      if c is not None and low <= c < high]
            if subset:
                correct = sum(1 for t, p in subset if p == t["label"])
                out.append((f"{low:.2f}-{min(high, 1.0):.2f}", correct, len(subset)))
        return out


def report(result: Result) -> None:
    hits, total = result.accuracy()
    clean_hits, clean_total = result.accuracy(skip_ambiguous=True)
    print(f"{result.name:<12} "
          f"{hits}/{total} = {hits / total:.0%} over all test tickets   |   "
          f"{clean_hits}/{clean_total} = {clean_hits / clean_total:.0%} "
          f"excluding the 5 ambiguous")
    if result.unparseable:
        print(f"{'':<12} {result.unparseable} reply(ies) could not be parsed")


def detail(result: Result) -> None:
    print()
    print(f"--- {result.name}, per class -------------------------------------")
    for label, (hits, total) in result.per_class().items():
        print(f"  {label:<16} {hits}/{total}")

    print()
    print(f"--- {result.name}, where it went wrong ---------------------------")
    wrong = [(t, p) for t, p, _ in result.rows if p != t["label"]]
    if not wrong:
        print("  nothing")
    for ticket, predicted in wrong:
        flag = "  (ambiguous)" if ticket.get("ambiguous") else ""
        print(f"  #{ticket['id']:<3} said {predicted or 'unparseable'!s:<16} "
              f"label says {ticket['label']}{flag}")
        print(f"       {ticket['text']}")

    calibration = result.calibration()
    if calibration:
        print()
        print(f"--- {result.name}, stated confidence vs observed accuracy ----")
        for bucket, correct, total in calibration:
            print(f"  said {bucket}   was right {correct}/{total} "
                  f"= {correct / total:.0%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--detail", action="store_true",
                        help="per-class breakdown, errors and calibration")
    args = parser.parse_args()

    test = ticket_set.TEST
    ambiguous = sum(1 for t in test if t.get("ambiguous"))
    print(f"model      {args.model}")
    print(f"test set   {len(test)} tickets, {ambiguous} of them ambiguous")
    print(f"train set  {len(ticket_set.TRAIN)} tickets, used by few-shot only")
    print("-" * 78)

    rules = Result("rules")
    for ticket in test:
        rules.add(ticket, baseline.classify(ticket["text"]), None)

    zero = Result("zero-shot")
    for ticket in test:
        label, confidence = parse(ask(args.model, ticket["text"], []))
        zero.add(ticket, label, confidence)

    shots = few_shot_messages()
    few = Result("few-shot")
    for ticket in test:
        label, confidence = parse(ask(args.model, ticket["text"], shots))
        few.add(ticket, label, confidence)

    for result in (rules, zero, few):
        report(result)

    if args.detail:
        for result in (rules, zero, few):
            detail(result)

    print()
    print("-" * 78)
    print("The gap between the two accuracy columns is the share of the score "
          "that\nmeasures agreement with one annotator rather than correctness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
