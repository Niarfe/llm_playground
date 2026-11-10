"""
Tests for the Phase 3 examples.

Nothing here talks to Ollama. These cover the parts that are ordinary code
and can be wrong quietly: the SQL guard, the result-set comparison, the
reference answers, and the train/test boundary.

The guard tests are the ones that matter most. It is a security control on
text a language model wrote, so "probably fine" is not a standard, and the
read-only connection is tested against the engine rather than taken on
trust.
"""

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data_science_examples"


def load(folder: str, name: str):
    """
    Load a module from a data_science_examples folder.

    The folder goes on sys.path first, because these modules import their
    siblings by bare name (`import questions`) the same way they do when run
    as scripts.
    """
    directory = DS / folder
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    path = directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{folder}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def sql_example():
    return load("text_2_sql", "run")


@pytest.fixture(scope="session")
def sql_questions():
    return load("text_2_sql", "questions")


@pytest.fixture(scope="session")
def db(tmp_path_factory, sql_example):
    """A read-only connection to a freshly built database, as run.py uses."""
    path = tmp_path_factory.mktemp("db") / "billing.db"
    sql_example.build_database(path)
    return sql_example.open_read_only(path)


@pytest.fixture(scope="session")
def classifier():
    return load("classification", "run")


@pytest.fixture(scope="session")
def tickets():
    return load("classification", "tickets")


@pytest.fixture(scope="session")
def baseline():
    return load("classification", "baseline")


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------

ACCEPTED = [
    "SELECT 1",
    "select id from invoices where status = 'paid';",
    "  SELECT * FROM invoices  ",
    "WITH p AS (SELECT * FROM payments) SELECT COUNT(*) FROM p",
    "-- a comment first\nSELECT 1",
    "/* block comment */ SELECT 1",
]

REJECTED = [
    "",
    "   ",
    "DROP TABLE invoices",
    "DELETE FROM invoices",
    "INSERT INTO invoices VALUES (1)",
    "UPDATE invoices SET amount_cents = 0",
    "ALTER TABLE invoices ADD COLUMN x TEXT",
    "PRAGMA writable_schema = 1",
    "ATTACH DATABASE '/tmp/x.db' AS x",
    "VACUUM",
    "SELECT 1; DROP TABLE invoices;",
    "SELECT 1; SELECT 2",
    "EXPLAIN SELECT 1",
    "select * from invoices; delete from invoices",
]


@pytest.mark.parametrize("sql", ACCEPTED)
def test_guard_accepts_read_only_queries(sql_example, sql):
    assert sql_example.guard(sql) is None, sql


@pytest.mark.parametrize("sql", REJECTED)
def test_guard_rejects_everything_else(sql_example, sql):
    assert sql_example.guard(sql) is not None, sql


def test_read_only_connection_refuses_writes(db):
    """
    The second defence, checked against the engine.

    The guard is a string check and string checks can be fooled. This is the
    one that cannot be talked around.
    """
    with pytest.raises(sqlite3.OperationalError):
        db.execute("DELETE FROM invoices")
    with pytest.raises(sqlite3.OperationalError):
        db.execute("DROP TABLE invoices")


# ---------------------------------------------------------------------------
# the reference answers
# ---------------------------------------------------------------------------

def test_every_reference_query_runs(db, sql_questions):
    """A broken reference would fail every model silently."""
    for question in sql_questions.QUESTIONS:
        if question["sql"] is None:
            continue
        db.execute(question["sql"]).fetchall()


def test_reference_answers_are_what_the_docstring_claims(db, sql_questions):
    by_text = {q["question"]: q for q in sql_questions.QUESTIONS}

    amount = db.execute(
        by_text["What is the total amount of invoice 4472?"]["sql"]
    ).fetchall()
    assert amount == [(112000,)]

    outstanding = db.execute(
        by_text["List the IDs of every invoice that is still outstanding."]["sql"]
    ).fetchall()
    assert sorted(r[0] for r in outstanding) == [4472, 4475, 4476, 4477, 4478]

    late = db.execute(
        by_text["Which invoices received a payment after their due date?"]["sql"]
    ).fetchall()
    assert sorted(r[0] for r in late) == [4466, 4469, 4473, 4475]


def test_exactly_one_question_is_unanswerable(sql_questions):
    unanswerable = [q for q in sql_questions.QUESTIONS if q["sql"] is None]
    assert len(unanswerable) == 1
    assert "dispute" in unanswerable[0]["question"]


def test_line_items_sum_to_invoice_amounts(db):
    """The property schema.sql claims. If it breaks, the data was edited."""
    mismatched = db.execute("""
        SELECT i.id FROM invoices i JOIN line_items l ON l.invoice_id = i.id
        GROUP BY i.id
        HAVING SUM(l.qty * l.unit_price_cents) != i.amount_cents
    """).fetchall()
    assert mismatched == []


def test_the_ambiguous_question_really_is_ambiguous(db, sql_questions):
    """
    Four readings, and they must not agree. If they ever did, the question
    would stop teaching anything and should be rewritten.
    """
    values = [db.execute(sql).fetchone()[0]
              for sql in sql_questions.AMBIGUITY_READINGS.values()]
    assert len(set(values)) == len(values)
    assert None in values, "the NULL-returning reading is the point of one of them"


# ---------------------------------------------------------------------------
# comparison and cleanup
# ---------------------------------------------------------------------------

def test_row_order_ignored_unless_the_question_asked_for_it(sql_example):
    a, b = [(1,), (2,)], [(2,), (1,)]
    assert sql_example.same_results(a, b, ordered=False)
    assert not sql_example.same_results(a, b, ordered=True)


def test_duplicate_rows_are_not_collapsed(sql_example):
    """Set comparison would call these equal. They are not."""
    assert not sql_example.same_results([(1,), (1,)], [(1,)], ordered=False)


def test_extra_columns_fail(sql_example):
    """Q5's failure mode, pinned so a future loosening is deliberate."""
    assert not sql_example.same_results([(1, "x")], [(1,)], ordered=False)


@pytest.mark.parametrize("reply,expected", [
    ("SELECT 1", "SELECT 1"),
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("Here you go:\n```sql\nSELECT 1\n```\nHope that helps", "SELECT 1"),
    ("  SELECT 1  ", "SELECT 1"),
])
def test_clean_extracts_sql(sql_example, reply, expected):
    assert sql_example.clean(reply) == expected


# ---------------------------------------------------------------------------
# classification: the train/test boundary
# ---------------------------------------------------------------------------

def test_no_ticket_is_in_both_splits(tickets):
    train = {t["id"] for t in tickets.TRAIN}
    test = {t["id"] for t in tickets.TEST}
    assert train & test == set()
    assert len(train) + len(test) == len(tickets.TICKETS)


def test_few_shot_prompt_contains_no_test_ticket(classifier, tickets):
    """
    The leak, tested directly.

    Sampling the examples from the whole set is the natural thing to write
    and it silently inflates the score, so this asserts the boundary at the
    place where it would break.
    """
    prompt = " ".join(m["content"] for m in classifier.few_shot_messages())
    for ticket in tickets.TEST:
        assert ticket["text"] not in prompt, ticket["id"]


def test_labels_are_valid_and_balanced(tickets):
    from collections import Counter
    assert all(t["label"] in tickets.LABELS for t in tickets.TICKETS)
    assert set(Counter(t["label"] for t in tickets.TEST).values()) == {8}


def test_ambiguous_tickets_are_all_in_the_test_split(tickets):
    """An ambiguous training example would teach the judgement call as fact."""
    assert not any(t.get("ambiguous") for t in tickets.TRAIN)
    assert sum(1 for t in tickets.TEST if t.get("ambiguous")) == 5


# ---------------------------------------------------------------------------
# classification: parsing and the baseline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reply,label,confidence", [
    ("billing|0.9", "billing", 0.9),
    ("bug|1", "bug", 1.0),
    ("  account_access | 0.75  ", "account_access", 0.75),
    ("FEATURE_REQUEST|0.5", "feature_request", 0.5),
    ("billing", "billing", None),
])
def test_parse_reads_label_and_confidence(classifier, reply, label, confidence):
    assert classifier.parse(reply) == (label, confidence)


@pytest.mark.parametrize("reply", [
    "",
    "I am not sure about this one.",
    "urgent|0.9",
    "none of the above",
])
def test_parse_refuses_to_guess(classifier, reply):
    """
    Unparseable must stay unparseable.

    Mapping a garbage reply onto a label hides the failure, and a hidden
    failure in the parsing layer is indistinguishable from a model that
    works.
    """
    assert classifier.parse(reply) == (None, None)


def test_baseline_always_returns_a_valid_label(baseline, tickets):
    for ticket in tickets.TICKETS:
        assert baseline.classify(ticket["text"]) in tickets.LABELS
    assert baseline.classify("") in tickets.LABELS


def test_baseline_beats_chance_but_is_not_suspiciously_good(baseline, tickets):
    """
    A guard against re-contaminating the baseline.

    75% on the measured run. If someone tunes the keyword lists against this
    test set the number climbs, and the whole point of the baseline is lost,
    so the upper bound is asserted as well as the lower one. Breaking the top
    of this range should prompt the question "how did I know to add that
    word?" rather than a celebration.
    """
    hits = sum(baseline.classify(t["text"]) == t["label"] for t in tickets.TEST)
    accuracy = hits / len(tickets.TEST)
    assert 0.25 < accuracy < 0.90, f"{hits}/{len(tickets.TEST)}"
