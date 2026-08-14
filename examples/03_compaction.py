"""
03 — Stop the conversation from outgrowing the context window.

THE PROBLEM
    Example 01 ends with a warning: history only grows. Every turn resends
    the entire transcript, so cost and latency climb turn over turn until
    you hit the context limit and the model starts dropping the oldest
    messages silently -- usually the ones holding the setup.

THE FIX
    Every N user turns, ask the model to compress everything except the
    last few turns into a short fact list, then replace the raw history
    with that summary. The prompt stays roughly constant-size instead of
    growing linearly.

        [system] + [50 old messages] + [4 recent]
     -> [system] + [1 summary]       + [4 recent]

WHAT YOU LOSE
    Detail. The summary is lossy and it is lossy in ways you do not control
    -- the model decides what mattered. Example 04 adds a fact store to
    recover specifics that compaction threw away.

WHY RAW HTTP HERE
    This example uses `requests` against Ollama's REST API instead of the
    `ollama` SDK, to show there is no magic in the SDK: it is JSON over
    localhost. Examples 01 and 02 use the SDK. Either is fine; pick one
    per project rather than mixing them like this repo originally did.

RUN IT
    python examples/03_compaction.py

    Talk for seven or more turns and watch the [compaction] notice fire.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "dolphin3"

SUMMARIZE_EVERY = 6  # compact once the history holds this many user turns
KEEP_LAST = 4        # always keep this many recent messages verbatim

SUMMARY_SYSTEM_PROMPT = (
    "You are compacting a conversation for reuse. Read the conversation "
    "below and produce a compact bullet list of durable facts, decisions, "
    "preferences, and open questions. Omit small talk and anything "
    "resolved or no longer relevant. Do not add commentary. Output only "
    "the bullet list."
)


def call_ollama(messages, model=MODEL):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def summarize(messages_to_compact):
    """Collapse a list of {role, content} messages into a fact-list string."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages_to_compact
    )
    return call_ollama(
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
    )


def maybe_compact(history):
    """
    history: list of {role, content}, NOT including the system prompt.
    Returns a possibly-compacted history.

    Note this is a pure function of its input -- no globals, no I/O beyond
    the summarizer call. That is what makes it easy to reason about and to
    lift into a real application later.
    """
    user_turns = sum(1 for m in history if m["role"] == "user")
    if user_turns < SUMMARIZE_EVERY or len(history) <= KEEP_LAST:
        return history

    to_compact = history[:-KEEP_LAST]
    to_keep = history[-KEEP_LAST:]

    print(f"\n[compaction] squashing {len(to_compact)} messages into a summary...")
    fact_summary = summarize(to_compact)
    print(f"[compaction] summary:\n{fact_summary}\n")

    return [
        {"role": "system", "content": f"Conversation summary so far:\n{fact_summary}"},
        *to_keep,
    ]


def chat_loop():
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    history = []  # everything except the system prompt

    print(f"Chatting with {MODEL} -- type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        # Compact BEFORE sending, so the outgoing prompt stays small.
        history = maybe_compact(history)

        reply = call_ollama([system_prompt] + history)
        history.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}")
        print(f"[history: {len(history)} messages]\n")


if __name__ == "__main__":
    chat_loop()
