# Notes

Things learned building the examples, including the parts that did not work.

## Ollama

**Two ways in, pick one.** The `ollama` Python SDK and plain HTTP against
`localhost:11434` do the same thing. The SDK returns objects
(`response.message.content`); raw HTTP returns dicts
(`response.json()["message"]["content"]`). This repo shows both on purpose —
examples 01/02/07/08 use the SDK, 03/04 use `requests`, 06 uses `httpx` because
it needs async streaming. In a real project, choose one and stay there. Mixing
them was the single messiest thing about this code before it was cleaned up.

**Streaming is newline-delimited JSON.** One object per token, so
`aiter_lines()` plus `json.loads` per line. Empty lines happen; skip them rather
than letting a `JSONDecodeError` kill the stream.

**Model names are inconsistent in list responses.** Depending on version the
field is `.model`, `["model"]`, or `["name"]`. Example 02 checks all three.

**Tool calling is not universal.** llama3.1 and qwen2.5 support it. Several
models will silently answer in prose instead of returning `tool_calls`, which
looks like a bug in your code and is not.

## Context and memory

**Compaction is the fix for unbounded history, and it is lossy in ways you do
not control.** The summarizer decides what was "durable". Anything it drops is
gone. Pairing it with a fact store (example 04) covers the gap, but only for
facts the extractor happened to notice.

**Keyword retrieval is worth trying before embeddings.** No embedding model, no
vector store, no per-turn cost. It fails in two directions, and the second one
is the surprise:

- *Misses:* "feline" will never match "cat".
- *False hits:* "How old is the feline?" **does** return the cat fact, because
  both contain "old". Score is divided by query length, so on a short query one
  incidental common word clears the threshold easily.

Both are pinned down in `tests/test_fact_memory.py`. Reach for embeddings when
you can name a real query this gets wrong, not before.

**Extraction is expensive, retrieval is cheap.** Run extraction at compaction
time only. Running it per turn doubles your model calls for almost no gain.

**Models are bad at "output only JSON".** They add fences, they add "Sure!".
Parse defensively and return empty on failure — losing a few facts beats
crashing the conversation.

## Text to speech on macOS

The short version: **use the `say` binary, not a Python TTS library.**

The long version is in `archive/pyttsx3/`. Six attempts at making `pyttsx3`
work, all failing in the same family of ways:

- The first syllable gets clipped.
- The first utterance plays, then everything after it is silent.
- Fixes involving `startLoop(False)` and manual `iterate()` deadlock against a
  chat loop.

Root cause: `pyttsx3` on macOS wraps a Cocoa API that expects to be driven from
the main thread and to own the event loop. Any interactive program already has
plans for its main thread, so the two fight. Going straight to `AppKit`'s
`NSSpeechSynthesizer` (`archive/pyttsx3/bulletproof.py`) did fix it — but by
then `subprocess` + `say` was simpler, had no dependency, and could not
deadlock.

**Two tricks that matter more than the engine choice:**

1. `[[slnc 500]]` — an Apple speech command for 500ms of silence, prepended to
   every utterance. CoreAudio needs a moment to wake up; without the pad it eats
   the first syllable. This alone fixed most of what looked like an engine
   problem.

2. Sanitize markdown first. Unsanitized, the engine reads asterisks and URLs
   aloud, which is unlistenable.

**For streaming, buffer into sentences and merge aggressively.** Speaking each
short fragment as it arrives spawns a `say` process per clause, and the gaps
between processes are audible as stutter. Holding fragments under ~40 characters
and merging everything currently queued into one `say` call sounds continuous.
The cost is a slightly longer wait before the first word. There is no setting
that wins both.

## Prompting

**Most of what feels like a model's personality is the system prompt.** Holding
one fixed while varying the other is the fastest way to learn what a model
actually contributes.

**Structure changes reasoning quality.** Forcing STATE / GOAL / MOVE / PREDICT /
OBSERVE noticeably helps small models on arithmetic word problems — each step
gets written where the next can use it. Committing to a bound *before*
calculating ("the result is under 48") gives the model something to contradict
itself against, which is a self-check it will not otherwise perform.

**But the self-check does not actually work.** In an observed run of example 08,
qwen2.5:7b-instruct produced:

```
PREDICT: The result is less than 48 + 24 + 36 - 48 = 56
OBSERVE: 48 + 24 + 36 = 108
Consistent
```

108 is not less than 56. It wrote "Consistent" because that token follows
OBSERVE in the pattern, not because it compared anything. It also wrote PREDICTs
that perform the calculation inside the prediction — the one thing the prompt
explicitly forbids.

The final answer was still correct. So the structure appears to help the
arithmetic while the verification step is theater. That is worth knowing before
building anything on top of a model grading its own output, and it is the
strongest argument for scoring these runs externally rather than trusting the
model's own "Consistent".

Example 08 shows the technique. Measuring it properly — fixed question sets,
repeated trials, scored output — is a different activity and needs its own
harness. A single side-by-side is an anecdote.

**Modelfiles vs system prompts.** A Modelfile bakes the prompt and parameters
into a named model (`ollama create`). Convenient for something used daily,
worse for experimenting, since changing a word means rebuilding. Files read at
runtime win while you are still learning what works.

## Repo history

This started as one flat sandbox with four threads tangled together — the
runner, compaction, TTS, and the structured-reasoning experiments. Commit
`f327e54` is that original state if anything here needs checking against it.
