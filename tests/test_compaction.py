"""
Tests for 04_compaction.py.

No Ollama required: the summarizer is monkeypatched, which is possible
precisely because maybe_compact() is a pure function of its input.
"""


def make_history(turns: int) -> list[dict]:
    """Build `turns` user/assistant pairs -- so 2 * turns messages."""
    history = []
    for i in range(turns):
        history.append({"role": "user", "content": f"question {i}"})
        history.append({"role": "assistant", "content": f"answer {i}"})
    return history


def test_short_history_is_untouched(compaction, monkeypatch):
    monkeypatch.setattr(compaction, "summarize", lambda _: "SHOULD NOT BE CALLED")

    history = make_history(1)  # 1 user turn, below the threshold
    assert compaction.maybe_compact(history) == history


def test_compaction_fires_at_threshold(compaction, monkeypatch):
    monkeypatch.setattr(compaction, "summarize", lambda _: "- a durable fact")

    history = make_history(compaction.SUMMARIZE_EVERY_N_USER_TURNS)
    result = compaction.maybe_compact(history)

    assert len(result) < len(history)
    assert result[0]["role"] == "system"
    assert "a durable fact" in result[0]["content"]


def test_recent_turns_survive_verbatim(compaction, monkeypatch):
    monkeypatch.setattr(compaction, "summarize", lambda _: "summary")

    history = make_history(compaction.SUMMARIZE_EVERY_N_USER_TURNS)
    expected_tail = history[-compaction.KEEP_LAST_N_MESSAGES:]

    result = compaction.maybe_compact(history)

    # summary + the last KEEP_LAST_N_MESSAGES messages, unchanged
    assert result[1:] == expected_tail
    assert len(result) == compaction.KEEP_LAST_N_MESSAGES + 1


def test_only_older_messages_are_summarized(compaction, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        compaction, "summarize", lambda msgs: seen.update(passed=msgs) or "summary"
    )

    history = make_history(compaction.SUMMARIZE_EVERY_N_USER_TURNS)
    compaction.maybe_compact(history)

    assert seen["passed"] == history[: -compaction.KEEP_LAST_N_MESSAGES]
    assert history[-1] not in seen["passed"]


def test_the_two_constants_use_different_units(compaction):
    """
    Documents a trap the names now defend against.

    SUMMARIZE_EVERY_N_USER_TURNS counts user turns; KEEP_LAST_N_MESSAGES
    slices messages. One turn is two messages, so "keep the last 4" keeps
    two exchanges, not four. The names carry their units for this reason --
    if either is ever renamed to something unitless, this test is the note
    explaining why that was a bad idea.
    """
    history = make_history(compaction.SUMMARIZE_EVERY_N_USER_TURNS)
    assert len(history) == compaction.SUMMARIZE_EVERY_N_USER_TURNS * 2

    kept = history[-compaction.KEEP_LAST_N_MESSAGES:]
    exchanges = sum(1 for m in kept if m["role"] == "user")
    assert exchanges == compaction.KEEP_LAST_N_MESSAGES // 2


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

    assert max(sizes) <= compaction.SUMMARIZE_EVERY_N_USER_TURNS * 2 + 2
