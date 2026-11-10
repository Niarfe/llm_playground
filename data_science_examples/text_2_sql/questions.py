"""
The question set, and the reference SQL each question is graded against.

Every reference query below was written and checked by hand against the 16
invoices in schema.sql. That is the whole basis for trusting the grades, so
if you change the data, re-check the references.

Fields:
    question   what a person would actually type
    sql        the reference query, or None if the question cannot be
               answered from this schema at all
    ordered    True when row order is part of the answer, because the
               question asked for a ranking
    teaches    one line on why this question is in the set

The questions are in rough order of difficulty, and the last two are not
difficulty at all. They are a different kind of problem.
"""

QUESTIONS = [
    # -- lookup ------------------------------------------------------------
    dict(
        question="What is the total amount of invoice 4472?",
        sql="SELECT amount_cents FROM invoices WHERE id = 4472;",
        ordered=False,
        teaches="The floor. One table, one row, one column. If this fails, "
                "nothing else is worth measuring.",
    ),

    # -- filter ------------------------------------------------------------
    dict(
        question="List the IDs of every invoice that is still outstanding.",
        sql="SELECT id FROM invoices WHERE status = 'outstanding';",
        ordered=False,
        teaches="Requires knowing that 'outstanding' is a literal in the "
                "status column, which the schema says but does not spell out "
                "unless the column comment is passed to the model.",
    ),

    # -- join --------------------------------------------------------------
    dict(
        question="Which customers have at least one outstanding invoice? "
                 "Give their names.",
        sql="""SELECT DISTINCT c.name
                 FROM customers c
                 JOIN invoices i ON i.customer_id = c.id
                WHERE i.status = 'outstanding';""",
        ordered=False,
        teaches="The first query that cannot be answered from one table. "
                "DISTINCT matters: Alder Freight would appear once here, but "
                "a customer with two outstanding invoices would duplicate.",
    ),

    # -- aggregate + group + order ----------------------------------------
    dict(
        question="What is the total outstanding amount for each customer? "
                 "Put the largest first.",
        sql="""SELECT c.name, SUM(i.amount_cents) AS total
                 FROM customers c
                 JOIN invoices i ON i.customer_id = c.id
                WHERE i.status = 'outstanding'
             GROUP BY c.id
             ORDER BY total DESC;""",
        ordered=True,
        teaches="GROUP BY plus ORDER BY, and the first question where row "
                "order is part of being right.",
    ),

    # -- derived condition, no such column --------------------------------
    dict(
        question="Which invoices received a payment after their due date?",
        sql="""SELECT DISTINCT i.id
                 FROM invoices i
                 JOIN payments p ON p.invoice_id = i.id
                WHERE p.paid_on > i.due;""",
        ordered=False,
        teaches="There is no 'late' or 'overdue' column. The condition has "
                "to be derived by comparing two dates across a join. A model "
                "that invents WHERE i.overdue = 1 fails loudly, which is the "
                "good kind of failure.",
    ),

    # -- ambiguous, and the reference is a choice -------------------------
    dict(
        question="How much revenue did we book in October 2026?",
        sql="""SELECT SUM(amount_cents)
                 FROM invoices
                WHERE issued BETWEEN '2026-10-01' AND '2026-10-31';""",
        ordered=False,
        teaches="AMBIGUOUS. 'Booked' could mean invoices issued in October, "
                "invoices due in October, or cash received in October. All "
                "three are defensible, all three are different numbers, and "
                "the reference picks one. Grading this question at all is a "
                "small act of pretending the question was clear.",
    ),

    # -- unanswerable -----------------------------------------------------
    dict(
        question="Which invoices are currently in dispute?",
        sql=None,
        ordered=False,
        teaches="UNANSWERABLE. There is no dispute column and no disputes "
                "table. The only correct response is to say so. A model that "
                "writes WHERE status = 'disputed' produces valid SQL that "
                "runs, returns zero rows, and reads as the welcome news that "
                "nothing is in dispute. Nothing about the output looks wrong.",
    ),
]

# Other readings of the ambiguous question, printed alongside the graded
# answer so the spread is visible rather than asserted.
AMBIGUITY_READINGS = {
    "invoices issued in October (the reference)":
        """SELECT SUM(amount_cents) FROM invoices
            WHERE issued BETWEEN '2026-10-01' AND '2026-10-31';""",
    "invoices due in October":
        """SELECT SUM(amount_cents) FROM invoices
            WHERE due BETWEEN '2026-10-01' AND '2026-10-31';""",
    "cash actually received in October":
        """SELECT SUM(amount_cents) FROM payments
            WHERE paid_on BETWEEN '2026-10-01' AND '2026-10-31';""",
    # Added after a run: qwen2.5-coder:7b produced this reading, which nobody
    # writing the question had thought of. It is defensible -- revenue you
    # have been paid for -- and on this data it matches no invoice at all,
    # so SUM returns NULL. Not zero. NULL.
    "paid invoices issued in October (a model's reading)":
        """SELECT SUM(amount_cents) FROM invoices
            WHERE issued BETWEEN '2026-10-01' AND '2026-10-31'
              AND status = 'paid';""",
}
