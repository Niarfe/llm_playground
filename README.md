# llm_playground

Small, self-contained examples for working with local LLMs through
[Ollama](https://ollama.com). Each file teaches exactly one idea and runs on
its own.

This is a **teaching repo**, not a library. There is no shared package to
import, and the examples deliberately repeat small amounts of code — a
twenty-line Ollama client copied four times is a fair price for being able to
read any single file top to bottom without chasing an import. Nothing here is
meant to be depended on; copy what is useful into your own code.

## Setup

```bash
make env && make update
```

Then, in a separate terminal:

```bash
ollama serve
```

Pull the models the examples use:

```bash
ollama pull dolphin3 && ollama pull llama3.1 && ollama pull qwen2.5:7b-instruct
```

## The examples

Read them in order. Each one starts from a problem the previous one leaves open.

| | | |
|---|---|---|
| 01 | [minimal_chat](examples/01_minimal_chat.py) | `generate()` vs `chat()`, and why the `messages` list is the whole game |
| 02 | [pick_model_and_prompt](examples/02_pick_model_and_prompt.py) | Swap models and system prompts at runtime; streaming output |
| 03 | [compaction](examples/03_compaction.py) | History grows forever — summarize old turns to keep the prompt bounded |
| 04 | [fact_memory](examples/04_fact_memory.py) | Compaction is lossy — a keyword-searchable fact store recovers specifics |
| 05 | [tts_say](examples/05_tts_say.py) | Speech on macOS via the `say` binary, plus the two tricks that make it bearable |
| 06 | [streaming_tts](examples/06_streaming_tts.py) | Speak while generating: sentence buffering and a queue-fed worker thread |
| 07 | [tool_calling](examples/07_tool_calling.py) | The agent loop, which is smaller than it sounds — and where the security boundary sits |
| 08 | [structured_reasoning](examples/08_structured_reasoning.py) | Changing output structure changes reasoning quality |

Supporting files: [`examples/prompts/`](examples/prompts) holds the system
prompts used by 02, and [`examples/modelfiles/`](examples/modelfiles) shows the
same personalities baked into Ollama Modelfiles instead.

## Running them

```bash
make run-01
```

Any example also runs directly:

```bash
env/bin/python examples/03_compaction.py
```

## Tests

```bash
make test
```

The tests cover the pure logic — compaction thresholds, retrieval scoring,
markdown sanitizing, sentence splitting, the tool-calling path boundary — and
need no running model. They double as documentation: `test_fact_memory.py`
records both why keyword retrieval misses synonyms *and* why it produces false
hits on short queries, which is the kind of thing you only notice by writing it
down.

## Also here

- **[notes.md](notes.md)** — what was learned, including what did not work.
- **[archive/pyttsx3/](archive/pyttsx3)** — a dead end, kept on purpose. Six
  attempts at Python-library TTS before abandoning it for the `say` binary.
  The failure sequence is more instructive than the working code.

## What this repo is not

It is not an application. The obvious next step — one program with `--tts`,
`--model`, `--compact` flags — is a different project with different goals, and
folding it in here would cost these files the clarity that is their only reason
to exist.
