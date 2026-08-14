# Archive: the pyttsx3 dead end

**Conclusion first: do not use `pyttsx3` for TTS on macOS in an interactive
program. Use the `say` binary — see [`examples/05_tts_say.py`](../../examples/05_tts_say.py).**

This folder is kept because the failure sequence is more instructive than the
fix. Nothing here is maintained or imported by anything else.

## The symptoms

Three distinct failures, which looked unrelated and were not:

1. **First-syllable clipping.** "Hello there" comes out "-llo there".
2. **One-shot audio.** The first utterance plays. Every one after it is
   silent, with no error raised.
3. **Deadlock.** Fixes for (2) hang the program instead.

## The root cause

`pyttsx3` on macOS wraps `NSSpeechSynthesizer`, a Cocoa API that expects to be
driven from the main thread and to own the run loop. Any interactive program
already has plans for its main thread. `runAndWait()` blocks it; `startLoop()`
wants to take it over. There is no arrangement that satisfies both the chat loop
and the speech engine.

The clipping is separate and simpler: CoreAudio takes a moment to open the
output device, and the beginning of the utterance is lost while it does.

## The attempts, in order

| File | Approach | Result |
|---|---|---|
| `pyttsx3_test.py` | `say()` + `runAndWait()` | Works exactly once |
| `list_voices.py` | Enumerate installed voices | Worked, still useful for reference |
| `complete_sol.py` | Fresh `pyttsx3.init()` per utterance, `[[slnc 600]]` pad | Fixed the silence and the clipping; spawns a whole engine per sentence |
| `thread_safe_for_ollama_example.py` | Generation on a worker thread, speech pinned to main | Correct threading model, still one-shot |
| `engine_running_example.py` | `startLoop(False)` + manual `iterate()` | Kept the channel warm, awkward to drive |
| `manual_loop_pumping.py` | Non-blocking queue polling + `iterate()` while busy | Closest working version; fragile |
| `bulletproof.py` | Drop `pyttsx3`, call `AppKit.NSSpeechSynthesizer` directly | **Actually worked.** One synth instance, `[[slnc 200]]` pad, queue-fed |

`ORIGINAL_NOTES.md` is the unedited notes file from while this was in progress,
including an `ffmpeg` incantation for holding the audio device open. That turned
out to be unnecessary once the silence padding was understood.

## Why `say` won anyway

`bulletproof.py` genuinely fixed it. It was still abandoned, because
`subprocess` + `say`:

- has no Python dependency at all
- cannot deadlock — each utterance is a separate process
- lets the OS own the audio session
- is about ten lines

The `AppKit` route is the better choice if you need fine control (rate changes
mid-sentence, interruption, precise callbacks). For a chat loop that needs to
read replies aloud, it is more machinery than the job requires.

## What carried forward

Two things from this folder survive in `examples/05` and `examples/06`:

- **`[[slnc N]]` padding.** Discovered here, and it turned out to be the fix for
  clipping regardless of which engine you use.
- **The queue-plus-worker-thread shape.** The threading model in
  `thread_safe_for_ollama_example.py` was right all along. Only the engine
  underneath it was wrong.
