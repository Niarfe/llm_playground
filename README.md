# llm_playground

Small, self-contained examples for building a local agentic AI setup with
[Ollama](https://ollama.com). Each file teaches one idea and runs on its own.

**The goal: by the end of the main line you will have built an agent loop and
understand every line of it.**

This is a **teaching repo**, not a library. There is no shared package to
import, and the examples deliberately repeat code -- a twenty-line block copied
across four files is a fair price for reading any single file top to bottom
without chasing an import. Copy what is useful into your own code.

That duplication is checked, not assumed: blocks marked `[unchanged from 02]`
are verified byte-identical by `tests/test_provenance.py`, so the copies cannot
silently drift.

## Prerequisites

Two things, and you almost certainly have the second already.

**1. Ollama** -- the local model runtime. Everything here talks to it.

Download from **[ollama.com/download](https://ollama.com/download)** (macOS,
Linux, Windows), or on a Mac with Homebrew:

```bash
brew install ollama
```

Check it worked:

```bash
ollama --version
```

Developed against Ollama 0.32.1. Anything recent should be fine.

**2. Python 3.9 or newer**, with `venv`. Check with `python3 --version`.
Developed on 3.14; nothing here needs a version that new.

`make` is used for convenience targets and ships with macOS (via Xcode command
line tools) and every Linux distribution. If you would rather not use it, every
target is a one-line Python command you can run directly -- see [the
makefile](makefile).

No API keys, no accounts, no network calls once the models are pulled.
Everything runs on your machine.

> **macOS note:** the two TTS examples under `extras/` shell out to the macOS
> `say` binary and will not work elsewhere. Nothing on the main line (01-09)
> depends on them.

## Setup

```bash
make env && make update && make models
```

Then, in a separate terminal:

```bash
ollama serve
```

`make models` pulls the two required models (~5.9 GB total):

| Model | Size | Why |
|---|---|---|
| `llama3.1` | 4.9 GB | The workhorse. Supports tool calling |
| `qwen2.5:1.5b-instruct` | 1.0 GB | Small and fast. Makes a correct single tool call 4/5 -- then falls apart on the multi-step loop in 08. That gap is the lesson |

`make models-optional` adds `qwen2.5:7b-instruct` and `deepseek-r1:8b`. Worth
pulling the first one: running 08 against all three models shows three distinct
behaviours -- qwen2.5:1.5b abandons its tools and invents an answer, llama3.1
batches dependent calls then recovers, qwen2.5:7b issues one call and waits.
That progression says more about model capability than any benchmark.

## The main line

Read them in order. Each starts from a problem the previous one leaves open.

| | | |
|---|---|---|
| 01 | [minimal_chat](examples/01_minimal_chat.py) | The model has no memory. You maintain the message list, always |
| 02 | [pick_model_and_prompt](examples/02_pick_model_and_prompt.py) | Vary model and system prompt without editing code |
| 03 | [temperature](examples/03_temperature.py) | The third control: how it samples. Reporting wants 0, generating does not |
| 04 | [compaction](examples/04_compaction.py) | The message list grows forever -- summarize old turns to bound it |
| 05 | [fact_memory](examples/05_fact_memory.py) | Compaction is lossy -- a searchable fact store recovers specifics |
| 06 | [tool_calling](examples/06_tool_calling.py) | The model requests, *your code* executes. One round trip |
| 07 | [retry_loop](examples/07_retry_loop.py) | A loop that resamples. Useful -- but not an agent |
| 08 | [agent_loop](examples/08_agent_loop.py) | **The centrepiece.** A loop where each result informs the next call. Run it twice -- `make run-08` and `make run-08-gated` |
| 09 | [branching_agent](examples/09_branching_agent.py) | *Experimental.* A task where what it reads decides which record it opens next. Scored by an independent checker -- `make run-09-all` |

The 07/08 pair is the point of the repo. They look almost identical in code and
differ in one thing: whether the context changes between iterations.

## Extras

Off the main line. Remove any of them and the agent loop still works -- that is
the test for what earned a number.

| | | |
|---|---|---|
| *fun* | [tts_say](examples/extras/tts_say.py) | Speech via the macOS `say` binary, and the two tricks that make it bearable |
| *fun* | [streaming_tts](examples/extras/streaming_tts.py) | Speak while generating: sentence buffering and a worker thread |
| *quality* | [structured_reasoning](examples/extras/structured_reasoning.py) | Output structure changes reasoning quality -- and self-checks don't work |

## Running

```bash
make run-01
```

Or directly: `env/bin/python examples/08_agent_loop.py`

Every interactive example takes the same commands: `/exit`, `/context`,
`/clear`. `/context` prints exactly what is about to be sent to the model --
in a teaching repo the internal state is the lesson, so it gets printed.

## Tests

```bash
make test
```

84 tests, no running model required. They cover the pure logic -- compaction
thresholds, retrieval scoring, schema validation, tool boundaries, markdown
sanitizing -- and double as documentation. Several record *limitations* rather
than asserting correctness: `test_loops.py` pins down why tool output phrasing
is load-bearing, and `test_fact_memory.py` records both why keyword retrieval
misses synonyms and why it produces false hits on short queries.

## Also here

- **[notes.md](notes.md)** -- what was learned, including what did not work.
- **[archive/pyttsx3/](archive/pyttsx3)** -- a dead end, kept on purpose. Six
  attempts at Python-library TTS before abandoning it for the `say` binary.

## What this repo is not

It is not an application. The obvious next step -- one program with `--tts`,
`--model`, `--compact` flags -- is a different project with different goals, and
folding it in here would cost these files the clarity that is their only reason
to exist.
