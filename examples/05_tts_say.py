"""
05 — Make the assistant talk, using the macOS `say` binary.

WHAT THIS TEACHES
    Text to speech on macOS in about thirty lines, with no Python audio
    library at all: shell out to `say`, the binary that ships with the OS.

WHY NOT A PYTHON TTS LIBRARY
    This repo tried pyttsx3 first and it went badly -- clipped first
    syllables, silence after the first utterance, thread deadlocks. See
    archive/pyttsx3/README.md for the full autopsy. The short version:
    pyttsx3 wraps a Cocoa API that must be driven from the main thread,
    which fights any chat loop.

    `say` sidesteps all of it. Each utterance is a fresh process, the OS
    handles the audio session, and there is nothing to deadlock.

THE TWO NON-OBVIOUS BITS
    1. SANITIZE. Raw model output is markdown. Unsanitized, `say` reads
       asterisks, backticks and URLs aloud, which is unbearable.

    2. PAD. `[[slnc 500]]` is an Apple speech command for 500ms of silence.
       CoreAudio takes a moment to wake up, and without the pad it eats the
       first syllable. This is the single highest-value trick here.

BLOCKING VS NOT
    subprocess.Popen returns immediately -- speech overlaps your next
    print, and two replies can talk over each other.
    subprocess.run waits -- clean sequencing, but your loop stalls until
    the sentence finishes.
    Neither is right for streaming. Example 06 solves it with a queue.

RUN IT
    python examples/05_tts_say.py
    say -v '?'        # list installed voices
"""

import re
import subprocess


class TextToSpeech:
    def __init__(self, voice: str = "Samantha", blocking: bool = True):
        self.enabled = True
        self.voice = voice
        self.blocking = blocking

    def sanitize(self, text: str) -> str:
        """Strip markdown so the speech engine does not read punctuation aloud."""
        text = re.sub(r"```[\s\S]*?```", " Code block omitted. ", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)      # inline code ticks
        text = re.sub(r"[*_]{1,3}", "", text)          # bold / italic markers
        text = re.sub(r"http[s]?://\S+", "link omitted", text)
        return text.strip()

    def speak(self, text: str) -> None:
        if not self.enabled or not text:
            return

        clean = self.sanitize(text)
        if not clean:
            return

        cmd = ["say"]
        if self.voice:
            cmd.extend(["-v", self.voice])

        # [[slnc 500]] = 500ms of silence, so CoreAudio is awake before
        # the first real syllable. Drop this and you will hear it.
        cmd.append(f"[[slnc 500]] {clean}")

        run = subprocess.run if self.blocking else subprocess.Popen
        try:
            run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[TTS] `say` not found -- this example is macOS only.")
            self.enabled = False


if __name__ == "__main__":
    tts = TextToSpeech(voice="Samantha")

    print("Speaking three sentences in order...\n")
    for line in [
        "This is the first sentence, and the padding keeps its opening intact.",
        "The second follows cleanly because the call blocks until the first finishes.",
        "Markdown like **bold** and `code` is stripped before it reaches the engine.",
    ]:
        print(f"  [speaking] {line}")
        tts.speak(line)

    print("\nNow the same text without sanitizing, for contrast:")
    raw = "Check **this** out at https://example.com for `details`."
    print(f"  [speaking raw] {raw}")
    subprocess.run(
        ["say", "-v", "Samantha", raw],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
