"""
Tests for 04_fact_memory.py.

The retrieval path is pure and needs no model at all. The extraction path
is tested against the malformed output that models actually produce.
"""

import pytest

FACTS = [
    "The user's cat is named Blue.",
    "Blue the cat is 4 years old.",
    "The project deadline is March 14th.",
    "The user prefers dark mode.",
]


# ---------- tokenize ----------

def test_tokenize_drops_stopwords_and_short_words(memory):
    assert memory.tokenize("What is the cat") == {"cat"}


def test_tokenize_is_case_insensitive(memory):
    assert memory.tokenize("BLUE") == memory.tokenize("blue")


def test_tokenize_keeps_apostrophes_and_digits(memory):
    tokens = memory.tokenize("user's 2024 deadline")
    assert "user's" in tokens
    assert "2024" in tokens


# ---------- retrieval ----------

def test_retrieves_the_matching_fact(memory):
    hits = memory.retrieve_relevant_facts("How old is Blue?", FACTS)
    assert "Blue the cat is 4 years old." in hits


def test_unrelated_query_retrieves_nothing(memory):
    assert memory.retrieve_relevant_facts("What is the weather in Oslo?", FACTS) == []


def test_respects_top_k(memory):
    hits = memory.retrieve_relevant_facts("cat Blue deadline mode", FACTS, top_k=2)
    assert len(hits) <= 2


def test_empty_query_returns_nothing(memory):
    assert memory.retrieve_relevant_facts("the a is", FACTS) == []


def test_known_limitation_synonyms_miss(memory):
    """
    Documents the tradeoff rather than asserting the code is perfect.
    Word overlap cannot match "feline" to "cat" -- that is the price of
    having no embedding model. If this ever starts passing a synonym,
    retrieval got smarter and the example's docstring needs updating.
    """
    assert memory.retrieve_relevant_facts("Describe the feline", FACTS) == []


def test_known_limitation_incidental_words_cause_false_hits(memory):
    """
    The flip side, and the more surprising failure mode.

    "How old is the feline?" DOES retrieve the cat fact -- not because it
    understood "feline", but because both contain the word "old". Scoring
    is divided by query length, so on a two-word query a single common
    word scores 0.5 and sails past the 0.15 threshold.

    Short queries are therefore easy to fool. This is worth knowing before
    trusting retrieval, and it is invisible until you write it down.
    """
    hits = memory.retrieve_relevant_facts("How old is the feline?", FACTS)
    assert "Blue the cat is 4 years old." in hits


# ---------- extraction (model output parsing) ----------

@pytest.mark.parametrize(
    "raw",
    [
        '["fact one", "fact two"]',
        '```json\n["fact one", "fact two"]\n```',
        '```\n["fact one", "fact two"]\n```',
    ],
)
def test_extraction_survives_markdown_fences(memory, monkeypatch, raw):
    monkeypatch.setattr(memory, "call_ollama", lambda *a, **k: raw)
    assert memory.extract_facts([]) == ["fact one", "fact two"]


def test_extraction_returns_empty_on_garbage(memory, monkeypatch):
    monkeypatch.setattr(memory, "call_ollama", lambda *a, **k: "Sure! Here you go:")
    assert memory.extract_facts([]) == []


def test_extraction_filters_non_strings(memory, monkeypatch):
    monkeypatch.setattr(memory, "call_ollama", lambda *a, **k: '["ok", 42, "", null]')
    assert memory.extract_facts([]) == ["ok"]


# ---------- store ----------

def test_corrupt_store_does_not_raise(memory, monkeypatch, tmp_path):
    bad = tmp_path / "facts.json"
    bad.write_text("{ this is not json")
    monkeypatch.setattr(memory, "FACTS_PATH", bad)
    assert memory.load_facts() == []


def test_missing_store_returns_empty(memory, monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "FACTS_PATH", tmp_path / "nope.json")
    assert memory.load_facts() == []


def test_save_then_load_roundtrips(memory, monkeypatch, tmp_path):
    path = tmp_path / "facts.json"
    monkeypatch.setattr(memory, "FACTS_PATH", path)
    memory.save_facts(["one", "two"])
    assert memory.load_facts() == ["one", "two"]
