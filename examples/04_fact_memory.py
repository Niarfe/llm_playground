"""
04 — Remember specifics that compaction threw away.

THE PROBLEM
    Example 03 keeps the prompt small by summarizing old turns. But a
    summary is lossy: you told it your daughter's birthday in turn 2, the
    summarizer decided that was not "durable", and now it is gone. Ask
    about it in turn 40 and you get "I don't recall that."

THE FIX
    A fact store, kept on disk, separate from the conversation:

      WRITE   when you compact, also ask the model to extract atomic facts
              and append them to facts.json
      READ    on every turn, pull the few facts that overlap with what the
              user just asked, and inject them as a system message

    Compaction shrinks what you send. The fact store recovers what
    compaction discarded. They are opposite operations and worth learning
    separately -- which is why they are separate files here.

WHY KEYWORD MATCHING, NOT EMBEDDINGS
    Word overlap is embarrassingly simple and it is free: no embedding
    model, no vector database, no extra call per turn. It fails on synonyms
    ("spouse" will not match "wife"). Start here anyway. Reach for
    embeddings when you can point at a real query that this misses.

    Retrieval is cheap; extraction is expensive. That is why extraction
    only runs at compaction time, never per turn.

RUN IT
    python examples/04_fact_memory.py

    Try: "My cat is named Blue and she is 4."  then  "How old is my cat?"
    Inspect facts.json between turns to see what was stored.
"""

import json
import re
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "dolphin3"

FACTS_PATH = Path("facts.json")
RETRIEVAL_TOP_K = 3
RETRIEVAL_MIN_SCORE = 0.15  # fraction of the query's words that must overlap

# Words too common to carry meaning. Without this, "what" matches everything.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "what", "who", "when", "where", "why", "how", "i", "you", "we",
    "it", "to", "of", "in", "on", "for", "and", "or", "that", "this",
    "my", "our", "your", "about", "with",
}

FACT_EXTRACTION_PROMPT = (
    "Read the conversation below. Extract a JSON array of short, atomic, "
    "self-contained facts worth remembering long-term (decisions, names, "
    "numbers, project details, stated preferences). Each element must be "
    "a single string. Skip anything trivial or already obvious. Output "
    "ONLY the JSON array, nothing else."
)


def call_ollama(messages, model=MODEL):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# ---------- the store ----------

def load_facts():
    if not FACTS_PATH.exists():
        return []
    try:
        return json.loads(FACTS_PATH.read_text())
    except json.JSONDecodeError:
        return []  # a corrupt store should not kill the chat


def save_facts(facts):
    FACTS_PATH.write_text(json.dumps(facts, indent=2))


# ---------- write path: extraction ----------

def extract_facts(messages):
    """
    Ask the model for a JSON list of atomic facts.

    Models are unreliable about "output ONLY JSON" -- they wrap it in
    markdown fences, or prepend "Sure!". Parsing is best-effort and
    failure returns [] rather than raising: losing a few facts is much
    better than crashing the conversation.
    """
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    raw = call_ollama(
        [
            {"role": "system", "content": FACT_EXTRACTION_PROMPT},
            {"role": "user", "content": transcript},
        ]
    )

    try:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        facts = json.loads(cleaned)
        return [f for f in facts if isinstance(f, str) and f.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


# ---------- read path: retrieval ----------

def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def retrieve_relevant_facts(query, facts, top_k=RETRIEVAL_TOP_K):
    """
    Score = (shared words) / (words in the query).

    Dividing by the query length, not the fact length, means a short
    pointed question needs only one good hit to score well.
    """
    query_words = tokenize(query)
    if not query_words:
        return []

    scored = []
    for fact in facts:
        fact_words = tokenize(fact)
        if not fact_words:
            continue
        score = len(query_words & fact_words) / len(query_words)
        if score >= RETRIEVAL_MIN_SCORE:
            scored.append((score, fact))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [fact for _, fact in scored[:top_k]]


# ---------- loop ----------

def chat_loop():
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    history = []

    print(f"Chatting with {MODEL} -- type 'quit' to exit.")
    print("Commands: /facts to dump the store, /remember to extract now.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        if user_input.lower() == "/facts":
            for fact in load_facts():
                print(f"  - {fact}")
            print()
            continue

        if user_input.lower() == "/remember":
            # In a real app this fires during compaction (example 03).
            # Here it is manual so the write path is easy to watch.
            new_facts = extract_facts(history)
            if new_facts:
                save_facts(load_facts() + new_facts)
                print(f"[stored {len(new_facts)} facts]\n")
            else:
                print("[nothing worth storing]\n")
            continue

        history.append({"role": "user", "content": user_input})

        hits = retrieve_relevant_facts(user_input, load_facts())

        messages = [system_prompt]
        if hits:
            print(f"[recalled: {hits}]")
            messages.append(
                {
                    "role": "system",
                    "content": "Relevant facts from earlier:\n"
                    + "\n".join(f"- {f}" for f in hits),
                }
            )
        messages.extend(history)

        reply = call_ollama(messages)
        history.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    chat_loop()
