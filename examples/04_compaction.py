"""
04 — Stop the conversation from outgrowing the context window.

MAIN POINT
    Every turn resends the entire transcript, so cost and latency climb
    turn over turn until you hit the context limit — at which point the
    model silently drops the oldest messages, usually the ones holding your
    setup.

    Fix: every N turns, compress everything except the last few into a short
    fact list, and send that instead.

        [system] + [50 old messages] + [4 recent]
     -> [system] + [1 summary]       + [4 recent]

    The prompt stays roughly constant-size instead of growing linearly.

ONE TURN IS TWO MESSAGES
    Worth stating because the counters below look wrong otherwise. Each
    exchange appends your message AND the model's, so message count climbs
    by two per turn. It is also why context fills about twice as fast as
    people estimate.

WHAT YOU LOSE
    Detail, and you do not choose which. The summarizer decides what
    mattered. 05 adds a fact store to recover specifics it discarded.

PREVIOUSLY
    01 ended on "this list only grows." This is that bill coming due.

NEXT
    05 recovers what compaction throws away.

RUN IT
    python examples/04_compaction.py

    Talk for four or more turns and watch [compaction] fire. Use /context
    before and after to see exactly what changed.
"""

import ollama

MODEL = "llama3.1"

SUMMARIZE_EVERY_N_USER_TURNS = 4
KEEP_LAST_N_MESSAGES = 4  # 4 messages == 2 exchanges. Different unit on purpose.

SUMMARY_SYSTEM_PROMPT = (
    "You are compacting a conversation for reuse. Read the conversation "
    "below and produce a compact bullet list of durable facts, decisions, "
    "preferences, and open questions. Omit small talk and anything "
    "resolved or no longer relevant. Do not add commentary. Output only "
    "the bullet list."
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
# TALKING TO OLLAMA                                              [new in 04]
##############################################################################
# temperature 0 for the summarizer. Summarizing is a reporting step, not a
# creative one -- see 03. A summary that varies between runs is a summary you
# cannot reason about.

def call_ollama(messages, temperature=0.0):
    response = ollama.chat(
        model=MODEL, messages=messages, options={"temperature": temperature}
    )
    return response.message.content


##############################################################################
# COMPACTION                                                     [new in 04]
##############################################################################
# maybe_compact is a pure function of its input: no globals, no I/O beyond the
# summarizer call. That is what makes it easy to reason about, easy to test
# (see tests/), and easy to lift into a real application later.

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
    """
    user_turns = sum(1 for m in history if m["role"] == "user")
    if (
        user_turns < SUMMARIZE_EVERY_N_USER_TURNS
        or len(history) <= KEEP_LAST_N_MESSAGES
    ):
        return history

    to_compact = history[:-KEEP_LAST_N_MESSAGES]
    to_keep = history[-KEEP_LAST_N_MESSAGES:]

    print(f"\n[compaction] squashing {len(to_compact)} messages into a summary")
    fact_summary = summarize(to_compact)
    print(f"[compaction] result:\n{fact_summary}\n")

    return [
        {"role": "system", "content": f"Conversation summary so far:\n{fact_summary}"},
        *to_keep,
    ]


##############################################################################
# THE LOOP                                                       [new in 04]
##############################################################################

def chat_loop():
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    history = []

    print(f"Chatting with {MODEL}")
    print(f"Commands: {COMMANDS_HELP}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_input:
            continue

        handled, history, should_exit = handle_command(
            user_input, system_prompt, history
        )
        if should_exit:
            return
        if handled:
            continue

        history.append({"role": "user", "content": user_input})

        # Compact BEFORE sending, so the outgoing prompt stays small.
        history = maybe_compact(history)

        reply = call_ollama([system_prompt] + history, temperature=0.8)
        history.append({"role": "assistant", "content": reply})

        turns = sum(1 for m in history if m["role"] == "user")
        print(f"\nAssistant: {reply}")
        print(f"[history: {len(history)} messages / {turns} user turns]\n")


if __name__ == "__main__":
    chat_loop()
