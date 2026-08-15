"""
05 — Remember specifics that compaction threw away.

MAIN POINT
    Compaction is lossy and you do not choose what it loses. You mentioned
    your cat's name in turn 2, the summarizer judged it not "durable", and
    it is gone. A fact store kept on disk, separate from the conversation,
    gets it back:

      WRITE   at compaction time, extract atomic facts to facts.json
      READ    each turn, inject the few that overlap with the question

    Compaction shrinks what you send. The fact store recovers what
    compaction discarded. Opposite operations — hence separate files.

PROVING IT ACTUALLY WORKS
    Easy to fool yourself here. Say "my cat is Blue", ask two turns later,
    get "Blue" — and conclude the fact store worked when the model simply
    read it off the still-present conversation.

    So this example prints whether the answer is still in the raw history.
    When it says NOT in history and a fact is injected anyway, the store is
    genuinely what answered. Use /forget to drop the history and see it.

WHY WORD OVERLAP AND NOT EMBEDDINGS
    No embedding model, no vector database, no per-turn cost. It fails in
    two directions, and the second is the surprise — see tests/. Start here;
    reach for embeddings when you can name a query this gets wrong.

PREVIOUSLY
    04 kept the prompt bounded by throwing detail away. This is the other half.

NEXT
    06 lets the model do something rather than just say something.

RUN IT
    python examples/05_fact_memory.py

    Try: "My cat is named Blue and she is 4."
         /remember          (extract to the store)
         /forget            (wipe the conversation, keep the store)
         "How old is my cat?"
"""

import json
import re
from pathlib import Path

import ollama

MODEL = "llama3.1"

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


##############################################################################
# COMMANDS                                                 [unchanged from 02]
##############################################################################
# One input line can mean two things: talk to the model, or control the
# program. The leading slash is what separates them -- without it there is no
# way to ask a model about the word "quit".
#
# Bare `quit` and `exit` are accepted too. The habit is strong and refusing
# them teaches nobody anything.

COMMANDS_HELP = "/exit  /context  /clear"


def handle_command(user_input, system_prompt, history):
    """
    Returns (handled, history, should_exit).

    `/context` prints exactly what the model is about to receive. In a
    teaching repo the internal state is the lesson, so it gets printed.
    """
    cmd = user_input.lower().strip()

    if cmd in {"/exit", "/quit", "quit", "exit"}:
        return True, history, True

    if cmd == "/context":
        messages = [system_prompt] + history
        print(f"\n[context] {len(messages)} messages going to the model:")
        for m in messages:
            body = " ".join(m["content"].split())
            print(f"  {m['role']:>9} | {body[:88]}")
        print()
        return True, history, False

    if cmd == "/clear":
        print("[cleared] history reset\n")
        return True, [], False

    return False, history, False


##############################################################################
# TALKING TO OLLAMA                                              [new in 05]
##############################################################################

def call_ollama(messages, temperature=0.0):
    response = ollama.chat(
        model=MODEL, messages=messages, options={"temperature": temperature}
    )
    return response.message.content


##############################################################################
# THE STORE — plain JSON on disk                                 [new in 05]
##############################################################################
# Deliberately boring. The point is that memory is a file you control, not a
# feature of the model.

def load_facts():
    if not FACTS_PATH.exists():
        return []
    try:
        return json.loads(FACTS_PATH.read_text())
    except json.JSONDecodeError:
        return []  # a corrupt store should not kill the chat


def save_facts(facts):
    FACTS_PATH.write_text(json.dumps(facts, indent=2))


##############################################################################
# WRITE PATH — extraction                                        [new in 05]
##############################################################################
# temperature 0: this is an extraction step, not a creative one (see 03).
# Models are unreliable about "output ONLY JSON" -- they add fences, they add
# "Sure!". Parsing is best-effort and failure returns [] rather than raising,
# because losing a few facts beats crashing the conversation.

def extract_facts(messages):
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


##############################################################################
# READ PATH — retrieval                                          [new in 05]
##############################################################################

def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def retrieve_relevant_facts(query, facts, top_k=RETRIEVAL_TOP_K):
    """
    Score = (shared words) / (words in the query).

    Dividing by query length, not fact length, means a short pointed question
    needs only one good hit to score well. It also means a short question can
    be fooled by one incidental shared word -- see tests/.
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


##############################################################################
# THE LOOP                                                       [new in 05]
##############################################################################

def chat_loop():
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    history = []

    print(f"Chatting with {MODEL}")
    print(f"Commands: {COMMANDS_HELP}  /facts  /remember  /forget\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "/facts":
            facts = load_facts()
            print(f"\n[store] {len(facts)} facts in {FACTS_PATH}")
            for fact in facts:
                print(f"  - {fact}")
            print()
            continue

        if cmd == "/remember":
            # In a real app this fires during compaction (04). Manual here so
            # the write path is easy to watch.
            new_facts = extract_facts(history)
            if new_facts:
                save_facts(load_facts() + new_facts)
                print(f"[stored] {len(new_facts)} facts")
                for fact in new_facts:
                    print(f"  + {fact}")
                print()
            else:
                print("[stored] nothing worth keeping\n")
            continue

        if cmd == "/forget":
            history = []
            print("[forgot] conversation wiped. The fact store is untouched.")
            print("         Now ask something only the store can answer.\n")
            continue

        handled, history, should_exit = handle_command(
            user_input, system_prompt, history
        )
        if should_exit:
            return
        if handled:
            continue

        history.append({"role": "user", "content": user_input})

        # --- the proof, per the header ---
        facts = load_facts()
        hits = retrieve_relevant_facts(user_input, facts)

        query_words = tokenize(user_input)
        history_text = " ".join(m["content"] for m in history[:-1]).lower()
        in_history = any(w in history_text for w in query_words)

        print(f"[context] {len(history)} messages; "
              f"question's keywords {'ARE' if in_history else 'are NOT'} in raw history")

        messages = [system_prompt]
        if hits:
            print(f"[recalled] injecting {len(hits)} fact(s) from the store:")
            for fact in hits:
                print(f"  > {fact}")
            messages.append(
                {
                    "role": "system",
                    "content": "Relevant facts from earlier:\n"
                    + "\n".join(f"- {f}" for f in hits),
                }
            )
        else:
            print("[recalled] nothing matched in the store")

        messages.extend(history)

        reply = call_ollama(messages, temperature=0.8)
        history.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    chat_loop()
