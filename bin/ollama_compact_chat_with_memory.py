"""
Ollama chat loop with:
  1. Rolling compaction (as before) to keep the live prompt small.
  2. A local fact store (facts.json) populated during compaction.
  3. Cheap keyword-based retrieval against that store on every turn,
     so questions about things that scrolled out of context can still
     get answered from local memory instead of "I don't recall that."

No embeddings, no extra model calls per turn -- retrieval is a plain
local lookup. Extraction (the expensive part) only runs when
compaction runs, same as before.

Requires: pip install requests
Assumes Ollama is running locally: `ollama serve`.
"""

import json
import re
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1"

SUMMARIZE_EVERY = 6
KEEP_LAST = 4

FACTS_PATH = Path("facts.json")
RETRIEVAL_TOP_K = 3
RETRIEVAL_MIN_SCORE = 0.15  # fraction of query words that must overlap

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "what", "who", "when", "where", "why", "how", "i", "you", "we",
    "it", "to", "of", "in", "on", "for", "and", "or", "that", "this",
    "my", "our", "your", "about", "with",
}

SUMMARY_SYSTEM_PROMPT = (
    "You are compacting a conversation for reuse. Read the conversation "
    "below and produce a compact bullet list of durable facts, decisions, "
    "preferences, and open questions. Omit small talk and anything "
    "resolved or no longer relevant. Do not add commentary. Output only "
    "the bullet list."
)

FACT_EXTRACTION_PROMPT = (
    "Read the conversation below. Extract a JSON array of short, atomic, "
    "self-contained facts worth remembering long-term (decisions, names, "
    "numbers, project details, stated preferences). Each element must be "
    "a single string. Skip anything trivial or already obvious. Output "
    "ONLY the JSON array, nothing else."
)


def call_ollama(messages, model=MODEL):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------- compaction (summary for the live prompt) ----------

def summarize(messages_to_compact):
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages_to_compact
    )
    summary_messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    return call_ollama(summary_messages)


# ---------- fact store (persistent, on disk) ----------

def load_facts():
    if FACTS_PATH.exists():
        return json.loads(FACTS_PATH.read_text())
    return []


def save_facts(facts):
    FACTS_PATH.write_text(json.dumps(facts, indent=2))


def extract_facts(messages_to_compact):
    """Ask the model for a JSON list of atomic facts. Best-effort parsing."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages_to_compact
    )
    extraction_messages = [
        {"role": "system", "content": FACT_EXTRACTION_PROMPT},
        {"role": "user", "content": transcript},
    ]
    raw = call_ollama(extraction_messages)
    try:
        # strip stray markdown fences if the model adds them
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        facts = json.loads(cleaned)
        return [f for f in facts if isinstance(f, str) and f.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def retrieve_relevant_facts(query, facts, top_k=RETRIEVAL_TOP_K):
    """Plain word-overlap scoring. No model call, no embeddings."""
    query_words = tokenize(query)
    if not query_words:
        return []

    scored = []
    for fact in facts:
        fact_words = tokenize(fact)
        if not fact_words:
            continue
        overlap = query_words & fact_words
        score = len(overlap) / len(query_words)
        if score >= RETRIEVAL_MIN_SCORE:
            scored.append((score, fact))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [fact for _, fact in scored[:top_k]]


# ---------- compaction trigger ----------

def maybe_compact(history):
    user_turns = sum(1 for m in history if m["role"] == "user")
    if user_turns < SUMMARIZE_EVERY or len(history) <= KEEP_LAST:
        return history

    to_compact = history[:-KEEP_LAST]
    to_keep = history[-KEEP_LAST:]

    fact_summary = summarize(to_compact)

    new_facts = extract_facts(to_compact)
    if new_facts:
        facts = load_facts()
        facts.extend(new_facts)
        save_facts(facts)

    compacted = [
        {"role": "system", "content": f"Conversation summary so far:\n{fact_summary}"}
    ]
    compacted.extend(to_keep)
    return compacted


# ---------- main loop ----------

def chat_loop():
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    history = []

    print("Chatting with", MODEL, "- type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user_input})
        history = maybe_compact(history)

        facts = load_facts()
        hits = retrieve_relevant_facts(user_input, facts)

        messages = [system_prompt]
        if hits:
            recall_note = "Relevant facts from earlier in this project:\n" + "\n".join(
                f"- {f}" for f in hits
            )
            messages.append({"role": "system", "content": recall_note})
        messages.extend(history)

        reply = call_ollama(messages)
        history.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    chat_loop()
