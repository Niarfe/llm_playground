"""
Tests for 05_tts_say.py and 06_streaming_tts.py.

Nothing here makes a sound: sanitizing and sentence-splitting are pure
string functions, which is most of what actually goes wrong in TTS.
"""

import pytest


# ---------- sanitizing (05) ----------

def test_strips_bold_and_italic_markers(tts):
    engine = tts.TextToSpeech()
    assert engine.sanitize("This is **bold** and _italic_") == "This is bold and italic"


def test_strips_inline_code_ticks_but_keeps_the_word(tts):
    engine = tts.TextToSpeech()
    assert engine.sanitize("Run `ollama serve` now") == "Run ollama serve now"


def test_replaces_code_blocks_wholesale(tts):
    engine = tts.TextToSpeech()
    result = engine.sanitize("Try this:\n```python\nx = 1\n```\nDone.")
    assert "x = 1" not in result
    assert "Code block omitted" in result


def test_replaces_urls(tts):
    engine = tts.TextToSpeech()
    result = engine.sanitize("See https://example.com/a/b?c=d for more")
    assert "https" not in result
    assert "link omitted" in result


def test_empty_and_whitespace_sanitize_to_empty(tts):
    engine = tts.TextToSpeech()
    assert engine.sanitize("   \n  ") == ""


def test_speak_is_a_noop_when_disabled(tts, monkeypatch):
    calls = []
    monkeypatch.setattr(tts.subprocess, "run", lambda *a, **k: calls.append(a))

    engine = tts.TextToSpeech()
    engine.enabled = False
    engine.speak("hello")

    assert calls == []


def test_speak_pads_with_silence(tts, monkeypatch):
    """The 500ms pad is what stops CoreAudio eating the first syllable."""
    captured = []
    monkeypatch.setattr(tts.subprocess, "run", lambda cmd, **k: captured.append(cmd))

    engine = tts.TextToSpeech()
    engine.speak("hello there")

    assert "[[slnc 500]]" in captured[0][-1]


# ---------- sentence buffering (06) ----------

def test_incomplete_sentence_is_held_back(streaming):
    speakable, remainder = streaming.take_completed_sentences("The quick brown fox")
    assert speakable == ""
    assert remainder == "The quick brown fox"


def test_complete_long_sentence_is_released(streaming):
    text = "This sentence is comfortably longer than the minimum buffer length. "
    speakable, remainder = streaming.take_completed_sentences(text)
    assert speakable.strip() == text.strip()
    assert remainder == ""


def test_short_sentence_is_held_to_avoid_stutter(streaming):
    """
    "Sure!" alone would spawn its own `say` process and sound clipped.
    Holding it lets it merge with whatever comes next.
    """
    speakable, remainder = streaming.take_completed_sentences("Sure! ")
    assert speakable == ""
    assert remainder == "Sure! "


def test_trailing_fragment_is_preserved(streaming):
    text = "Here is a sentence long enough to be spoken on its own. And a par"
    speakable, remainder = streaming.take_completed_sentences(text)
    assert speakable.strip().endswith("own.")
    assert remainder == "And a par"


@pytest.mark.parametrize("punct", [".", "!", "?", "..."])
def test_all_sentence_terminators_are_recognized(streaming, punct):
    text = f"A sentence of sufficient length to clear the buffer minimum{punct} "
    speakable, _ = streaming.take_completed_sentences(text)
    assert speakable != ""


def test_no_text_is_lost_across_repeated_calls(streaming):
    """Feed a stream token by token; everything must come out exactly once."""
    full = (
        "The first sentence is long enough to be released on its own. "
        "The second one is also long enough to be released separately. "
        "Trailing fragment"
    )

    buffer = ""
    spoken = []
    for char in full:
        buffer += char
        speakable, buffer = streaming.take_completed_sentences(buffer)
        if speakable:
            spoken.append(speakable)

    assert "".join(spoken) + buffer == full
