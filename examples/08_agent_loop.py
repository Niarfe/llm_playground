"""
08 — The agent loop. This is the centrepiece.

MAIN POINT
    An agent loop is a while-loop in which each tool result becomes part of
    the next question. The model cannot know the answer up front; it has to
    gather information, and what it does next depends on what came back.

        while not done and turns_left:
            ask the model, with everything learned so far
            if it wants a tool -> run it, append the result, loop
            else                -> it has answered, stop

    That is the whole thing. Every "AI agent" you have read about is this,
    plus error handling and better tools.

WHY THIS TASK
    "Which example file is the longest, and how many lines is it?"

    It CANNOT be answered in one round trip. The model does not know what
    files exist, so it must call list_files first. It cannot count lines of
    files it has not named, so count_lines calls depend on that result.
    Then it compares. Roughly eight to ten calls, each genuinely needing the
    last one's output.

CONTRAST WITH 07
    07 asked the same question every attempt and hoped for a luckier sample.
    Here the message list grows with every result — watch the [turn N]
    counter and the message count climb together. Progress comes from
    accumulated knowledge, not variance.

    That growth is also why compaction (04) stops being theoretical. A long
    agent run fills a context window fast.

WHERE DO THE TOOL ARGUMENTS COME FROM?
    The most common question about this file, so: they come from the model,
    and nothing else produces them.

    The model does not "call" anything. It emits text. When the chat
    template shows it your tool definitions, a model trained for this emits
    something shaped like

        {"name": "count_lines", "parameters": {"file_name": "hello.py"}}

    Ollama parses that into `response.message.tool_calls`, the loop below
    unwraps it into `args`, and `tool(**args)` splats it into your real
    Python function. The filename is a token the model predicted, the same
    way it predicts any other word.

    Which is why it can invent `script1.py` out of nothing, and why it
    sometimes names a tool that does not exist at all. The first turn of
    every run prints the raw reply so you can see this rather than take it
    on faith.

    "Some models support tool calling" means some models were fine-tuned to
    emit that shape reliably. It is a learned output format, not a feature
    wired into the runtime.

WHAT YOU WILL ACTUALLY SEE
    The first turn usually goes badly, and that is the interesting part.
    llama3.1 typically invents filenames — script1.py, script2.py — that
    were never in the list. The tool refuses each one, the loop notes the
    repeated list_files calls, and by turn 2 the model has recovered and
    counts the real files.

    Do not read that as the example misbehaving. A single round trip that
    guessed wrong would simply be wrong, permanently. A loop gets to see
    "File not found", and correct itself. Recovery from its own mistakes is
    a property the loop provides and one-shot calling cannot.

    It is also why tool error messages deserve care: "File not found:
    script1.py" is what the model reads to figure out it went wrong.

TRY IT WITH THE SMALL MODEL
    Change MODEL to "qwen2.5:1.5b-instruct" and run it again.

    In 06 that model made a correct single tool call 4 times out of 5 —
    basically as good as llama3.1. Here it calls list_files once, then
    stops using the tools and starts writing plausible-looking file
    contents from imagination. Two turns, zero files actually counted,
    confidently wrong.

    That contrast is the most useful thing in this file after the loop
    itself. "Supports tool calling" is a capability flag both models
    declare. Sustaining a multi-step loop — keeping track of what it has
    learned, and continuing to use tools rather than inventing answers —
    is a separate and much harder capability that no flag reports.

THE BUDGET IS NOT DECORATION
    MAX_TURNS exists because models genuinely do get stuck: re-calling the
    same tool with the same arguments, or forgetting they already have what
    they need. Without a budget that is an infinite loop against a paid or
    slow endpoint. Watch for repeated calls in the output — you will see it.

PREVIOUSLY
    06 made a single tool call. 07 wrapped repetition around a call without
    accumulating anything.

NEXT
    Nothing on the main line — this is where the sequence lands. See
    examples/extras/ for optional directions, including giving it a voice.

RUN IT
    python examples/08_agent_loop.py
"""

from pathlib import Path

from ollama import chat

MODEL = "llama3.1"
MAX_TURNS = 12

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

QUESTION = (
    "Which Python script has the most lines, and how many lines does it "
    "have? Use the tools to find out."
)

# The loop alone is not enough, and this line is the evidence.
#
# Without this system prompt, llama3.1 called list_files, counted ONE file,
# invented three filenames that were never in the list, and confidently
# answered "04_compaction.py, 200 lines" -- wrong; the answer is
# 05_fact_memory.py at 301. qwen2.5:7b did better but still stopped early.
# With it, both get the right answer.
#
# The mechanism gives the model the ABILITY to gather information. It does
# not make it thorough. Process instructions are how you get thoroughness,
# and they belong in the system prompt, not buried in the question.
SYSTEM_PROMPT = (
    "You are a careful assistant. When asked to compare files, you MUST call "
    "count_lines for EVERY filename returned by list_files, one at a time, "
    "before answering. Never invent a filename. Only use names that "
    "list_files actually returned."
)


##############################################################################
# THE TOOLS                                                      [new in 08]
##############################################################################
# Two tools, deliberately narrow. Neither can answer the question alone --
# that is the point. list_files tells you what exists; count_lines needs a
# name it could only have learned from list_files.
#
# Same boundary discipline as 06: resolve, then check containment.

def list_files() -> str:
    """
    List every Python script available. Call this first — you cannot know
    the filenames otherwise.
    """
    names = sorted(p.name for p in SCRIPTS_DIR.glob("*.py"))
    if not names:
        return "(no python files found)"

    # Tool output is prompt text too. A bare newline-separated list gets
    # skimmed and half-ignored -- llama3.1 read one and then invented three
    # filenames that were never in it. Stating the count and enumerating
    # them in a sentence made that stop. Same lesson as the docstring in 06.
    return (
        f"There are exactly {len(names)} scripts: "
        + ", ".join(names)
        + ". You must count the lines of every one of these before answering."
    )


def count_lines(file_name: str) -> str:
    """
    Count the lines in one Python script.

    Args:
      file_name: Bare filename only, e.g. "hello.py". Must be a name that
        list_files returned. Do not include any directory path.
    """
    path = (SCRIPTS_DIR / file_name).resolve()

    if not str(path).startswith(str(SCRIPTS_DIR)) or path.suffix != ".py":
        return f"Refused: {file_name} is not a Python script in the scripts directory."
    if not path.exists():
        return f"File not found: {file_name}"

    return f"{file_name} has {len(path.read_text().splitlines())} lines"


TOOLS = {"list_files": list_files, "count_lines": count_lines}


##############################################################################
# THE LOOP                                                       [new in 08]
##############################################################################
# Compare this to 07's retry_loop. The shape is nearly identical -- a while,
# a budget, a call, a branch. The difference is one line: here the result is
# appended to `messages`, so the next iteration asks a better-informed
# question. In 07, `messages` never changed.

def agent_loop(question: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    print(f"[user] {question}\n")

    seen_calls = set()

    for turn in range(1, MAX_TURNS + 1):
        response = chat(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.values()),
            options={"temperature": 0},  # deciding and reporting, not creating
        )
        messages.append(response.message)

        # On the first turn only, show the unprocessed reply. Everything below
        # reads from tidy parsed fields, which hides where the values come
        # from -- and "where does the filename come from?" is THE question
        # about this example.
        #
        # The model did not call anything. It emitted TEXT that looked like
        # {"name": "count_lines", "parameters": {"file_name": "script1.py"}},
        # and Ollama parsed that into .tool_calls. The filename is a token the
        # model predicted, exactly like any other word it writes. That is why
        # it can invent one -- and why it sometimes names a tool that does not
        # exist, which is why line ~195 uses TOOLS.get() instead of TOOLS[].
        if turn == 1:
            print("[raw] the model's unparsed reply, before we interpret it:")
            print(f"      content   : {response.message.content!r}")
            for c in response.message.tool_calls or []:
                print(f"      tool_call : {c.function.name}"
                      f" args={dict(c.function.arguments)!r}")
            print("      ^ those args are model output. Nothing else produced them.\n")

        if not response.message.tool_calls:
            print(f"\n[turn {turn}] no tool requested -- the model is answering\n")
            print(f"[answer] {response.message.content}")
            print(f"\n[done] {turn} turns, {len(messages)} messages in context")
            return

        for call in response.message.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)
            signature = f"{name}({args})"

            # Not enforcement -- just visibility. Repetition is the classic
            # way these loops stall, and it is worth seeing when it happens.
            repeat = " <-- REPEAT" if signature in seen_calls else ""
            seen_calls.add(signature)

            print(f"[turn {turn:>2}] {signature}{repeat}")

            tool = TOOLS.get(name)
            result = tool(**args) if tool else f"Unknown tool: {name}"

            first_line = result.split("\n")[0]
            extra = f" (+{len(result.splitlines()) - 1} more lines)" if "\n" in result else ""
            print(f"          -> {first_line}{extra}")

            messages.append({"role": "tool", "content": result})

        print(f"          [context now {len(messages)} messages]")

    print(f"\n[gave up] hit the {MAX_TURNS}-turn budget without an answer.")
    print("This is the failure a budget exists to bound. Without it, this")
    print("loop would keep going until you killed it.")


if __name__ == "__main__":
    agent_loop(QUESTION)
