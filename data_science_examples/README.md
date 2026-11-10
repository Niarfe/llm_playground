# Data science examples

Phase 3 of the course: the places where LLMs are moving into work that was
already somebody's job.

`examples/` is a numbered reading order, because each file there depends on
the one before it. This folder is not that. Each example here is a complete
task, self-contained, and the order you take them in is up to whoever is
teaching. So they are named rather than numbered, one folder each, and
numbers can go on later if a sequence turns out to exist.

## What is here

| folder | task | how it is graded | measured 2026-11-10 |
|---|---|---|---|
| [`text_2_sql/`](text_2_sql/) | a question in English becomes a SQL query | run both queries, compare result sets | 4/7, `qwen2.5-coder:7b` |
| [`classification/`](classification/) | route support tickets into four labels | labels typed by a person | 75% rules, 88% zero-shot, 97% few-shot, `llama3.1` |

Each number is one run on one machine. The `MEASURED` section at the top of
each `run.py` says what the run actually showed, which in both cases is more
interesting than the score.

## Running them

Each needs `ollama serve` in another terminal.

```
make run-text2sql           # qwen2.5-coder:7b
make run-classify           # llama3.1
```

Both take `--model`, so you can swap in anything you have pulled and compare.
Both print what they measured rather than asserting it.

## The thread running through both

These two were picked first because they sit at opposite ends of one
question: **how do you know the output is right?**

`text_2_sql` has the cleanest answer available anywhere in this repo.
Generate the query, run it, run the reference query, compare the rows. No
judge model, no rubric, no partial credit. The grader is a database and a
database has no opinion.

`classification` has no such luxury. The grader is a column of labels
somebody typed, five of the thirty-two test tickets have a real claim on two
labels, and the accuracy number therefore includes an unknown amount of
"agrees with one annotator's judgement calls". It reports the score twice,
with and without those five, so the size of that unknown is visible.

That contrast is the actual lesson, and it transfers. Before building
anything on top of a model, find out which of those two situations you are
in. Most real tasks are the second one, and most people report them as
though they were the first.

## Two things both examples refuse to do

**Neither one lets a model write its own test data.** The invoices in
`text_2_sql/schema.sql` and the tickets in `classification/tickets.py` were
typed by a person, slowly. The reason is in the header of each file: if a
model generates the data and is then evaluated on it, part of the score is
measuring how well it agrees with itself. Generator and subject share a
prior, the eval confirms the prior, and the number comes out high and means
nothing.

`classification/baseline.py` documents a live instance of this that was not
a model's fault at all. An early version of the keyword baseline scored 31
out of 32, seven points above where it honestly lands, because the person
who wrote the keywords had just finished typing the tickets and reached for
their phrasing from memory. Nothing was tuned, no train/test boundary was
crossed, and the contamination arrived anyway.

**Neither one trusts the model's own confidence.** `classification` prints
stated confidence against observed accuracy at that confidence, which is the
only way to find out whether the number means anything.

It was written expecting to show that the confidence is decoration. On the
measured run, zero-shot prompting was decently calibrated instead: 71%
accurate where it said 0.7-0.9, 96% where it said 0.9+. The expectation was
wrong and the file says so.

Few-shot prompting then flattened the signal completely, reporting 0.9+ on
thirty of thirty-two tickets — because all eight demonstrations in the
prompt were hard-coded to `0.95` and the model copied the number along with
the label. Nine points of accuracy bought with the calibration. That defect
is still in the code, flagged at the line that causes it, because the
measurement is the evidence and the evidence is worth more to a student than
the fix.
