# Notes

Things learned building the examples, including the parts that did not work.

## Ollama

**Two ways in, pick one.** The `ollama` Python SDK and plain HTTP against
`localhost:11434` do the same thing. The SDK returns objects
(`response.message.content`); raw HTTP returns dicts
(`response.json()["message"]["content"]`). The main line uses the SDK
throughout; only `extras/streaming_tts.py` drops to `httpx`, because it needs
async streaming.

An earlier version of this repo mixed them — SDK in some examples, `requests`
in others — and that was the single messiest thing about it. Pick one and stay
there.

**Streaming is newline-delimited JSON.** One object per token, so
`aiter_lines()` plus `json.loads` per line. Empty lines happen; skip them rather
than letting a `JSONDecodeError` kill the stream.

**Model names are inconsistent in list responses.** Depending on version the
field is `.model`, `["model"]`, or `["name"]`. Example 02 checks all three.

**Tool calling is not universal.** llama3.1 and qwen2.5:7b support it;
qwen2.5:1.5b mostly does not. A model without it answers in prose instead of
returning `tool_calls`, which looks like a bug in your code and is not.

**Keep passing `tools=` on every call in the exchange.** Omit it on the
follow-up call — the one that reads the tool result — and the model replies as
though the tool never ran. The schema is part of the chat template, so without
it there is no rule for rendering a `"tool"` role message and the result is
silently dropped. The tool ran, the output is in `messages`, and the model
cannot see it. Cost me an hour.

**Tool docstrings are prompt text.** The SDK builds the schema the model sees
from the signature and docstring. A docstring reading "Execute a local Python
script" got both llama3.1 and qwen2.5 calling it with `/path/to/hello.py` — a
placeholder. Adding two constraint sentences ("Bare filename only… do not
invent a placeholder path") took it to 6/6 correct across both models.

**Tool *output* is prompt text too.** Same lesson, other direction. A
newline-separated file listing got skimmed: llama3.1 read one entry, invented
three filenames that were never in it, and answered from partial data. Changing
the return to "There are exactly 4 scripts: a.py, b.py… You must count every
one before answering" fixed it. What comes back from a tool is read by a
language model, so it should be written for one.

## Loops

**Retry loops and agent loops look almost identical and are not the same
thing.** Both are a `while`, a budget, a call, a check. The difference:

> Does the context change between iterations?
>
> **Retry** — same input, resampled output. Progress comes from variance.
> **Agent** — input grows with each result. Progress comes from knowledge.

A retry loop is rejection sampling: generate, validate, resample. Legitimate
and widely used — schema-valid output, flaky networks, best-of-N with a scorer.
It works when each attempt is an independent draw with a decent success rate.

It does not *make progress*. Asking an 8B model to solve a sudoku and checking
the answer burns 100 attempts and solves nothing, because independent draws
from a distribution that never contains the answer never produce it. Retrying
is not thinking.

**A retry loop at temperature 0 is an infinite loop.** Every attempt is
byte-identical, so if the first fails they all fail identically. Retry loops
*require* variance to function. Agent loops do not — 08 runs at temperature 0
and still progresses, because the input changes even though the sampling
doesn't.

**The loop is necessary but not sufficient — you also have to specify the
process.** With tools and a loop but no system prompt, llama3.1 called
`list_files`, counted one file, invented three filenames, and confidently
answered wrong. A system prompt saying "call count_lines for EVERY filename
returned, one at a time, before answering" fixed it on both models tested. The
mechanism grants the *ability* to gather information; it does not produce
thoroughness.

**Error recovery is the loop's real payoff, and it only shows up when things go
wrong.** Watching 08, the first turn usually goes badly — invented filenames,
repeated `list_files` calls. Then it reads "File not found: script1.py" and
corrects itself. A one-shot call that guessed wrong is wrong permanently; a
loop gets to see the error. This is also why tool error messages deserve care:
they are what the model reads to work out that it went wrong.

**Budgets are not decoration.** Models re-call the same tool with the same
arguments, or forget they already have the answer. Without a turn limit that is
an infinite loop against a slow or paid endpoint. Print a turn counter and flag
repeated calls — you will see it happen.

## Context and memory

**Compaction is the fix for unbounded history, and it is lossy in ways you do
not control.** The summarizer decides what was "durable". Anything it drops is
gone. Pairing it with a fact store (example 05) covers the gap, but only for
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

**But the self-check does not actually work.** In an observed run,
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

extras/structured_reasoning.py shows the technique. Measuring it properly — fixed question sets,
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

## Sampling

**Temperature is per call, not per app.** One program should use different
values at different steps. Reporting, extraction, and classification want 0.
Only genuinely generative work wants more.

This is not academic. The final call in `06_tool_calling.py` reports a fact
already sitting in `messages`; at the default temperature llama3.1 sometimes
narrated it as speculation — "since you didn't provide the script, I assume it
prints…" — while reciting the correct contents. Setting temperature 0 made
three consecutive runs byte-identical.

A Modelfile `PARAMETER temperature` bakes a default into the model;
`options={"temperature": N}` overrides it for a single call.

**Temperature 0 is not "better".** Same setting, opposite verdict depending on
the job: identical answers are correct behaviour for a factual question and a
failure for a creative one. `03_temperature.py` runs both so the contrast is
visible rather than asserted.
