"""
Tests for 07_retry_loop.py and 08_agent_loop.py.

The validator in 07 and the tools in 08 are pure functions, so the logic that
matters is testable without a model. What cannot be tested here is whether the
model behaves -- that is what running the examples is for.
"""

import json

import pytest


##############################################################################
# 07 -- the validator that gives the retry loop something to reject
##############################################################################

VALID = {
    "title": "A Book",
    "author": "Someone",
    "year": 1975,
    "tags": ["a", "b", "c"],
}


def test_accepts_a_well_formed_object(retry):
    ok, value = retry.validate(json.dumps(VALID))
    assert ok, value
    assert value["year"] == 1975


def test_rejects_wrong_tag_count(retry):
    """
    A constraint the model can satisfy but often does not. Without these the
    model passes almost every time and the retry loop never retries, which
    demonstrates nothing.
    """
    ok, reason = retry.validate(json.dumps(dict(VALID, tags=["a", "b"])))
    assert not ok
    assert "3" in reason


def test_rejects_year_outside_range(retry):
    ok, reason = retry.validate(json.dumps(dict(VALID, year=2024)))
    assert not ok
    assert "2024" in reason


def test_rejects_markdown_fences(retry):
    """The most common real failure -- models love wrapping JSON in fences."""
    ok, reason = retry.validate("```json\n" + json.dumps(VALID) + "\n```")
    assert not ok
    assert "not valid json" in reason.lower()


def test_rejects_chatty_preamble(retry):
    ok, reason = retry.validate("Sure! Here you go:\n" + json.dumps(VALID))
    assert not ok


@pytest.mark.parametrize("missing", ["title", "author", "year", "tags"])
def test_rejects_missing_keys(retry, missing):
    payload = {k: v for k, v in VALID.items() if k != missing}
    ok, reason = retry.validate(json.dumps(payload))
    assert not ok
    assert missing in reason


def test_rejects_wrong_types(retry):
    payload = dict(VALID, year="1999")
    ok, reason = retry.validate(json.dumps(payload))
    assert not ok
    assert "year" in reason


def test_rejects_booleans_masquerading_as_int(retry):
    """bool is a subclass of int in Python, so this needs an explicit guard."""
    ok, reason = retry.validate(json.dumps(dict(VALID, year=True)))
    assert not ok


def test_rejects_non_string_tags(retry):
    ok, reason = retry.validate(json.dumps(dict(VALID, tags=["ok", 3])))
    assert not ok
    assert "tags" in reason


def test_rejects_a_bare_array(retry):
    ok, reason = retry.validate("[1, 2, 3]")
    assert not ok
    assert "object" in reason


##############################################################################
# 08 -- the tools, and their boundaries
##############################################################################

def test_list_files_names_every_script(agent):
    listing = agent.list_files()
    for expected in ["hello.py", "greet.py", "add_numbers.py", "countdown.py"]:
        assert expected in listing


def test_list_files_states_the_count(agent):
    """
    Phrasing is load-bearing here, not cosmetic.

    A bare newline-separated list got skimmed: llama3.1 read one entry and
    then invented three filenames. Stating the count and enumerating them in
    a sentence stopped that. Tool output is prompt text.
    """
    listing = agent.list_files()
    assert "exactly 4 scripts" in listing


def test_count_lines_reports_a_real_count(agent):
    assert "13 lines" in agent.count_lines("add_numbers.py")


def test_count_lines_refuses_escape(agent):
    assert "refused" in agent.count_lines("../08_agent_loop.py").lower()


def test_count_lines_refuses_non_python(agent):
    assert "refused" in agent.count_lines("../../README.md").lower()


def test_count_lines_reports_missing_file(agent):
    """
    The model DOES invent filenames -- this message is what it reads to
    work out that it went wrong, so it has to be clear.
    """
    result = agent.count_lines("script1.py")
    assert "not found" in result.lower()
    assert "script1.py" in result


def test_the_longest_script_is_unambiguous(agent):
    """
    The example's answer must have exactly one right answer. If a fixture
    script grows and ties add_numbers.py, the demo becomes ambiguous and
    this test is the early warning.
    """
    counts = {}
    for name in ["hello.py", "greet.py", "add_numbers.py", "countdown.py"]:
        counts[name] = int(agent.count_lines(name).split()[-2])

    ranked = sorted(counts.values(), reverse=True)
    assert ranked[0] > ranked[1], f"no clear longest script: {counts}"
    assert max(counts, key=counts.get) == "add_numbers.py"
