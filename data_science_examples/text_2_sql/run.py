"""
TEXT TO SQL -- and a verifier that cannot be argued with.

WHERE THIS ATTACHES
    05a_rag.py asked which invoice was paid and got the wrong one. Semantic
    search put 4472 above 4471 by 0.758 to 0.756, because "Invoice 4471 was
    paid on the 2nd" and "Invoice 4472 is still outstanding" are nearly the
    same sentence in embedding space. The identifier, the only part that
    mattered, contributed almost nothing to the distance.

    The fix is not a better embedding model. The fix is to stop using
    similarity for a question that has an exact answer. The same two invoices
    are in this database, and here the question is answered by a query.

MAIN POINT
    This is the one data science task on the list where grading is not a
    matter of opinion.

        generate SQL from the question
        execute it
        execute the reference query
        compare the result sets

    No judge model. No rubric. No partial credit for effort. Either the rows
    match or they do not, and nothing about the model's confidence,
    fluency or formatting enters into it.

    Hold onto how rare that is. Summarization, extraction, classification
    against fuzzy labels -- all of those need either a human or another model
    to say whether the output was good, and the moment a model grades a model
    you have to start worrying about whether they share a blind spot. Here
    the grader is a database. It has no opinion and no prior.

WHY STRING COMPARISON WOULD BE WRONG
    The naive version of this grades the SQL text against the reference text.
    That measures the wrong thing. These two are the same query:

        SELECT id FROM invoices WHERE status = 'outstanding'
        SELECT invoices.id FROM invoices WHERE 'outstanding' = invoices.status

    and these two are not, though they differ by one character:

        WHERE p.paid_on >  i.due
        WHERE p.paid_on >= i.due

    Comparing text would mark the first pair wrong and might well mark the
    second pair right. Executing both and comparing results gets both cases
    correct, because it grades what the query DOES. The cost is that you need
    a database and trustworthy reference answers, which is why schema.sql is
    small enough to check by hand.

THE SEVEN QUESTIONS
    Five of them get harder in the ordinary way: lookup, filter, join,
    group-and-order, then a condition that has to be derived because the
    column it asks about does not exist.

    The last two are not about difficulty.

      Q6 is AMBIGUOUS. "How much revenue did we book in October 2026?"
         Invoices issued in October: 346,250. Invoices due in October:
         499,500. Cash actually received in October: 112,500. A 4.4x spread,
         and all three are honest readings of the same English sentence. The
         reference picks one. The script prints all three, because the
         interesting output here is not the grade but the gap between the
         grade and the other two defensible answers.

      Q7 is UNANSWERABLE. "Which invoices are currently in dispute?" There is
         no dispute column and no disputes table. The correct response is to
         say so.

         This is the question to spend time on. A model that writes

             SELECT id FROM invoices WHERE status = 'disputed'

         has produced valid SQL. It parses, it runs, it returns zero rows,
         and zero rows reads as the welcome news that nothing is in dispute.
         Every signal available downstream says the pipeline worked. You
         would have to go and read the schema yourself to find out that the
         answer was manufactured, and nobody reads the schema when the number
         looks fine.

         Q6 and Q7 are the same failure in two costumes: output that is
         structurally valid and semantically fabricated. The result set
         comparison catches Q7 because we told it the right answer was a
         refusal. It cannot catch Q6 at all, because the reference is itself
         a guess at what was meant. A verifier only checks the question you
         encoded, and the question you encoded may not be the question you
         were asked.

THE SANDBOX, WHICH IS NOT OPTIONAL
    This script executes text a language model wrote. That is the thing the
    secure sandboxing week is about, and here it is in the most innocuous
    looking task on the list.

    Two defences, both cheap:

      1. Every connection used for generated SQL is opened read-only, with
         SQLite's mode=ro URI. DROP TABLE fails at the database, not at our
         good intentions.
      2. A guard rejects anything that is not a single SELECT or WITH
         statement before it is ever handed to the database.

    Belt and braces, on purpose. The guard is a string check and string
    checks can be fooled; mode=ro is enforced by the engine. Neither one is
    sufficient and together they are adequate. Note also that the database is
    rebuilt from schema.sql into a temp file on every run, so the worst case
    is a wasted run.

MEASURED
    qwen2.5-coder:7b, temperature 0, 2026-11-10, M1 Mac. 4 of 7.

    Not a claim about the model in general. Seven questions, one run, one
    machine. What makes it worth writing down is that all three failures
    failed in a different way, and none of the three looked like a failure
    from the outside.

      Q5  THE LOGIC WAS RIGHT. The join and the date comparison were exactly
          the reference's, and it returned the same four invoices. It also
          returned three extra columns nobody asked for, so the result sets
          do not match and the verifier says no.

          That is the right call and it is also the verifier's weak spot. In
          a pipeline, extra columns break whatever consumes the rows, so this
          is a real defect. But "wrong shape" and "wrong answer" are not the
          same kind of wrong, and a single accuracy number hides which one
          you got. This is a known limitation of execution-accuracy grading
          generally, not a quirk of this file.

      Q6  IT FOUND A FOURTH READING. The question was already ambiguous
          three ways; the model added `AND status = 'paid'`, which is a
          perfectly defensible fifth-grade-English reading of "revenue we
          booked" and which nobody writing the question had considered.

          No invoice issued in October has been paid, so SUM returned NULL.
          Not an error. Not zero. NULL, which a report renders as a blank or
          a dash or a 0 depending on the chart library, and which reads as
          "no revenue in October" to everyone who sees it.

      Q7  IT SUBSTITUTED A CONCEPT IT HAD. Asked which invoices were in
          dispute, with no dispute column anywhere, it returned

              SELECT * FROM invoices WHERE status = 'outstanding'

          Five rows of entirely real invoices. Correct IDs, correct amounts,
          correct customers. Every value in that table is true. The answer is
          still manufactured, because "outstanding" is not "in dispute" and
          nothing in the output says the swap happened.

          This is worse than the invented-column failure the question was
          written to catch. An invented column raises a database error. A
          substituted concept returns clean data and a plausible number of
          rows, and the only way to catch it is to already know the schema
          well enough to know the question could not be answered.

    AND NOTE WHAT --repair BUYS HERE: nothing. The repair pass feeds SQL
    errors back to the model for a second try, and there were zero errors in
    this run. All three failures produced valid SQL that executed cleanly. A
    retry loop keyed on exceptions is blind to every failure mode above,
    which is a useful thing to have measured before trusting one.

RUN
    make run-text2sql
    make run-text2sql-repair     # feed SQL errors back for one retry

    python data_science_examples/text_2_sql/run.py --model llama3.1
    python data_science_examples/text_2_sql/run.py --verbose
"""

import argparse
import pathlib
import re
import sqlite3
import sys
import tempfile
from collections import Counter

import ollama

import questions as question_set

HERE = pathlib.Path(__file__).parent
SCHEMA = HERE / "schema.sql"

DEFAULT_MODEL = "qwen2.5-coder:7b"

REFUSAL = "CANNOT ANSWER"

SYSTEM_PROMPT = f"""\
You translate questions about a SQLite database into SQL.

Rules:
- Reply with one SQL statement and nothing else. No prose, no code fences,
  no explanation.
- Use only SELECT. The database is read only.
- If the schema does not contain the information needed to answer, reply with
  exactly `{REFUSAL}: ` followed by one short sentence saying what is missing.
  Do not guess at a column or a value that is not in the schema.

The schema:
"""


# ---------------------------------------------------------------------------
# the database
# ---------------------------------------------------------------------------

def build_database(path: pathlib.Path) -> None:
    """Build the database from schema.sql. Read-write, used once, then closed."""
    con = sqlite3.connect(path)
    con.executescript(SCHEMA.read_text())
    con.commit()
    con.close()


def open_read_only(path: pathlib.Path) -> sqlite3.Connection:
    """The only connection generated SQL ever touches."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def schema_text(con: sqlite3.Connection) -> str:
    """
    The CREATE TABLE statements, read out of the database itself.

    Taken from sqlite_master rather than from schema.sql, for two reasons.
    The seed data is not included, so the model cannot read the answers off
    the INSERT statements. And a hand-maintained copy of the schema in the
    prompt is one more thing that can drift out of date; this one cannot.
    """
    rows = con.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL"
    ).fetchall()
    return "\n\n".join(r[0].strip() + ";" for r in rows)


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------

WRITE_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|"
    r"DETACH|PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)


def guard(sql: str) -> str | None:
    """
    Reject anything that is not a single read-only statement.

    Returns None when the SQL is acceptable, or a string saying why not.
    This runs before the database sees the text. It is the cheaper of the two
    defences and the easier one to fool, which is why it is not the only one.
    """
    stripped = re.sub(r"--[^\n]*", " ", sql)
    stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.DOTALL).strip()

    if not stripped:
        return "empty statement"
    if stripped.count(";") > 1 or (stripped.count(";") == 1
                                   and not stripped.rstrip().endswith(";")):
        return "more than one statement"
    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        return "does not start with SELECT or WITH"
    if WRITE_WORDS.search(stripped):
        return f"contains a write keyword: {WRITE_WORDS.search(stripped).group(1)}"
    return None


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def clean(reply: str) -> str:
    """
    Pull the SQL out of whatever the model wrapped it in.

    Models put SQL in code fences no matter how firmly you ask them not to.
    That is not a failure of the model, it is what the training data looks
    like. Strip it and move on rather than counting it as an error -- the
    thing being measured is whether the query is right.
    """
    match = FENCE.search(reply)
    if match:
        return match.group(1).strip()
    return reply.strip()


def ask(model: str, schema: str, question: str,
        previous: tuple[str, str] | None = None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + schema},
        {"role": "user", "content": question},
    ]
    if previous:
        bad_sql, error = previous
        messages.append({"role": "assistant", "content": bad_sql})
        messages.append({
            "role": "user",
            "content": f"That failed with: {error}\nReply with corrected SQL only.",
        })
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0, "num_ctx": 4096},
    )
    return clean(response["message"]["content"])


# ---------------------------------------------------------------------------
# the verifier
# ---------------------------------------------------------------------------

def run_query(con: sqlite3.Connection, sql: str) -> tuple[list | None, str | None]:
    try:
        return con.execute(sql).fetchall(), None
    except sqlite3.Error as exc:
        return None, f"{type(exc).__name__}: {exc}"


def same_results(got: list, want: list, ordered: bool) -> bool:
    """
    Compare two result sets.

    Row order is ignored unless the question asked for a ranking, because
    SELECT without ORDER BY makes no promise about order and grading it would
    punish a correct query for an accident.

    Column order is NOT ignored, and neither is column count. A query that
    returns the right answer plus three extra columns was not asked for that,
    and in a pipeline the extra columns break whatever consumes them.
    """
    if ordered:
        return got == want
    return Counter(map(repr, got)) == Counter(map(repr, want))


def grade(con, generated: str, question: dict) -> tuple[str, str]:
    """Returns (verdict, detail). Verdict is one of pass / fail / error."""
    reference = question["sql"]

    # The unanswerable question. Correct behaviour is a refusal, so any SQL
    # at all is wrong here, however well formed.
    if reference is None:
        if generated.upper().startswith(REFUSAL):
            return "pass", "declined, correctly"
        return "fail", "answered a question the schema cannot answer"

    if generated.upper().startswith(REFUSAL):
        return "fail", "declined a question that was answerable"

    why = guard(generated)
    if why:
        return "error", f"blocked by the guard: {why}"

    got, err = run_query(con, generated)
    if err:
        return "error", err

    want, ref_err = run_query(con, reference)
    if ref_err:                                   # a bug in questions.py
        raise RuntimeError(f"the REFERENCE query failed: {ref_err}")

    if same_results(got, want, question["ordered"]):
        return "pass", f"{len(got)} row(s) match"
    return "fail", f"got {got!r}, reference gives {want!r}"


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repair", action="store_true",
                        help="on a SQL error, show the model the error once "
                             "and let it try again")
    parser.add_argument("--verbose", action="store_true",
                        help="print the generated SQL for every question")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "billing.db"
        build_database(path)
        con = open_read_only(path)
        schema = schema_text(con)

        print(f"model     {args.model}")
        print(f"questions {len(question_set.QUESTIONS)}")
        print(f"repair    {'on' if args.repair else 'off'}")
        print("-" * 72)

        tally = Counter()
        for n, question in enumerate(question_set.QUESTIONS, 1):
            sql = ask(args.model, schema, question["question"])
            verdict, detail = grade(con, sql, question)

            if verdict == "error" and args.repair:
                sql = ask(args.model, schema, question["question"],
                          previous=(sql, detail))
                verdict, detail = grade(con, sql, question)
                detail = f"after repair: {detail}"

            tally[verdict] += 1
            mark = {"pass": "PASS", "fail": "FAIL", "error": " ERR"}[verdict]
            print(f"{mark}  Q{n}  {question['question']}")
            if args.verbose or verdict != "pass":
                print(f"         sql: {' '.join(sql.split())}")
            print(f"         {detail}")
            print()

        print("-" * 72)
        total = sum(tally.values())
        print(f"{tally['pass']}/{total} pass   "
              f"{tally['fail']} wrong answers   "
              f"{tally['error']} did not run")

        # The ambiguous question, shown rather than asserted.
        print()
        print("Q6 asked how much revenue was booked in October 2026.")
        print("Three honest readings of that sentence:")
        for label, ref in question_set.AMBIGUITY_READINGS.items():
            value = con.execute(ref).fetchone()[0]
            shown = f"{value:,}" if value is not None else "NULL"
            print(f"   {shown:>10}   {label}")
        print()
        print("One of those was graded. The others were not wrong.")
        print("Note the NULL. SUM over no rows is not zero, and a dashboard")
        print("that renders it as a blank or a 0 has just told you revenue")
        print("was nothing when the truth is that nothing matched.")

    return 0 if tally["fail"] == 0 and tally["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
