"""
Minimal conversation compaction for a local Ollama chat loop.

Idea:
- Keep a rolling `messages` list like any chat client.
- Every N user turns, ask the model to compress everything older than
  the last KEEP_LAST turns into a compact fact list.
- Replace the raw history with: [system] + [summary] + [last KEEP_LAST turns].
- This keeps the prompt sent to Ollama roughly constant-size instead of
  growing linearly with conversation length.

Requires: pip install requests
Assumes Ollama is running locally: `ollama serve` (default port 11434).
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1"  # swap for whatever you've pulled

SUMMARIZE_EVERY = 6   # trigger compaction every N user turns
KEEP_LAST = 4         # always keep the most recent N messages verbatim

SUMMARY_SYSTEM_PROMPT = (
    "You are compacting a conversation for reuse. Read the conversation "
    "below and produce a compact bullet list of durable facts, decisions, "
    "preferences, and open questions. Omit small talk and anything "
    "resolved or no longer relevant. Do not add commentary. Output only "
    "the bullet list."
)


def call_ollama(messages, model=MODEL):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def summarize(messages_to_compact):
    """Collapse a list of {role, content} messages into a fact-list string."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages_to_compact
    )
    summary_messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    return call_ollama(summary_messages)


def maybe_compact(history, system_prompt):
    """
    history: list of {role, content} dicts, NOT including the system prompt.
    Returns a possibly-compacted history.
    """
    user_turns = sum(1 for m in history if m["role"] == "user")
    if user_turns < SUMMARIZE_EVERY or len(history) <= KEEP_LAST:
        return history

    to_compact = history[:-KEEP_LAST]
    to_keep = history[-KEEP_LAST:]

    fact_summary = summarize(to_compact)

    compacted = [
        {"role": "system", "content": f"Conversation summary so far:\n{fact_summary}"}
    ]
    compacted.extend(to_keep)
    return compacted


def chat_loop():
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    history = []  # everything except the system prompt

    print("Chatting with", MODEL, "- type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user_input})

        # Compact BEFORE sending, so the outgoing prompt stays small.
        history = maybe_compact(history, system_prompt)

        messages = [system_prompt] + history
        reply = call_ollama(messages)

        history.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    chat_loop()
