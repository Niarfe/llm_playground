"""
Tests for 03_compaction.py.

No Ollama required: the summarizer is monkeypatched, which is possible
precisely because maybe_compact() is a pure function of its input.
"""


def make_history(turns: int) -> list[dict]:
    """Build `turns` user/assistant pairs."""
    history = []
    for i in range(turns):
        history.append({"role": "user", "content": f"question {i}"})
        history.append({"role": "assistant", "content": f"answer {i}"})
    return history


def test_short_history_is_untouched(compaction, monkeypatch):
    monkeypatch.setattr(compaction, "summarize", lambda _: "SHOULD NOT BE CALLED")

    history = make_history(2)  # 2 user turns, below SUMMARIZE_EVERY
    assert compaction.maybe_compact(history) == history


def test_compaction_fires_at_threshold(compaction, monkeypatch):
    monkeypatch.setattr(compaction, "summarize", lambda _: "- a durable fact")

    history = make_history(compaction.SUMMARIZE_EVERY)
    result = compaction.maybe_compact(history)

    assert len(result) < len(history)
    assert result[0]["role"] == "system"
    assert "a durable fact" in result[0]["content"]


def test_recent_turns_survive_verbatim(compaction, monkeypatch):
    monkeypatch.setattr(compaction, "summarize", lambda _: "summary")

    history = make_history(compaction.SUMMARIZE_EVERY)
    expected_tail = history[-compaction.KEEP_LAST:]

    result = compaction.maybe_compact(history)

    # summary + the last KEEP_LAST messages, unchanged
    assert result[1:] == expected_tail
    assert len(result) == compaction.KEEP_LAST + 1


def test_only_older_messages_are_summarized(compaction, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        compaction, "summarize", lambda msgs: seen.update(passed=msgs) or "summary"
    )

    history = make_history(compaction.SUMMARIZE_EVERY)
    compaction.maybe_compact(history)

    assert seen["passed"] == history[: -compaction.KEEP_LAST]
    assert history[-1] not in seen["passed"]


def test_repeated_compaction_stays_bounded(compaction, monkeypatch):
    """The point of the whole exercise: history must not grow without bound."""
    monkeypatch.setattr(compaction, "summarize", lambda _: "summary")

    history = []
    sizes = []
    for i in range(40):
        history.append({"role": "user", "content": f"q{i}"})
        history = compaction.maybe_compact(history)
        history.append({"role": "assistant", "content": f"a{i}"})
        sizes.append(len(history))

    assert max(sizes) <= compaction.SUMMARIZE_EVERY * 2 + 2
