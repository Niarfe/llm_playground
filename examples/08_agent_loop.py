"""
08 -- The agent loop. This is the centrepiece.

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
    Here the message list grows with every result -- watch the [turn N]
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

    Consequence worth taking seriously: the model chooses the tool NAME and
    the ARGUMENTS, and both are just predicted tokens. It will name tools
    that do not exist and pass arguments that do not fit. `tool(**args)`
    with no guard is a crash waiting to happen -- this file used to have
    one, and a real run killed it with
    list_files(file_name='script1.py'). Validate at the dispatch, and hand
    failures back as tool results so the loop can recover.

WHAT YOU WILL ACTUALLY SEE, AND WHY IT IS NOT A HALLUCINATION
    Turn 1 emits something like this, all in ONE reply:

        list_files()
        count_lines(file_name='script1.py')
        count_lines(file_name='script2.py')

    It is tempting to call script1.py a hallucination -- the model ignoring
    filenames it was given. It is not, and the distinction matters.

    At that moment the message list is [system, question]. list_files has
    not run. NOTHING has run. The model has no possible way to know the real
    filenames, so it does the only thing it can: it plans the whole
    sequence, using placeholders for values it does not have yet.

    Measured over 5 runs, llama3.1 does this 5 times out of 5. It never
    emits list_files() alone and waits. The placeholders drift between runs
    -- script1.py, file1.py, hello.py, world.py -- which is what
    placeholder-generation looks like.

    So the real limitation is not invention. It is that the model does not
    know it should STOP and wait for a result it depends on. The loop is
    what saves it: the placeholders get refused, and on turn 2 -- now
    holding the real list -- it uses real names.

    The printed output makes this hard to see, because the loop executes
    that batch one call at a time and it reads as though the model saw each
    result before making the next call. It did not.

    IS THIS NORMAL? NO -- IT IS llama3.1 BEING WEAK
    Emitting several tool calls at once is a real feature, and correct when
    the calls are INDEPENDENT: "get the weather in these three cities" is
    three calls that can run in any order. These are not independent.
    count_lines needs what list_files returns. Batching them is incoherent.

    Swap MODEL to "qwen2.5:7b-instruct" and watch a model do it properly:

        qwen2.5:7b-instruct   1 call on turn 1, every run. Waits.
        llama3.1              9 calls on turn 1, every run. Speculates.

    Both reach the right answer here -- llama3.1 only because the loop lets
    it recover. Three models, three behaviours, and running all three
    teaches more than any explanation:

        qwen2.5:1.5b   abandons the tools on turn 2, invents an answer
        llama3.1       batches dependent calls, then recovers
        qwen2.5:7b     one call, waits for the result, proceeds

    DO NOT "FIX" THIS BY TRUNCATING THE BATCH
    The obvious mitigation is to execute only the first call per turn and
    discard the speculative rest. Measured, 3 runs each:

        llama3.1, execute all         3/3 correct, 3 turns
        llama3.1, execute first only  3/3 WRONG,   6 turns

    Throwing away calls the model asked for leaves it confused about what
    happened. Executing everything it requested -- and letting the failures
    come back as tool results -- works better than second-guessing it.

THE FAILURE MODE THIS EXAMPLE CANNOT DETECT
    The loop stops when `tool_calls` comes back empty. That is not the same
    as the model being finished. Three different situations arrive looking
    identical:

      1. It genuinely answered.
      2. It gave up and started inventing (see the small-model note below).
      3. It DID ask for a tool, but wrote prose first --

           'The tool returned: add_numbers.py, countdown.py...
            I will now count each one.
            {"name": "count_lines", "parameters": {...}}'

         -- and Ollama's parser, which expects the JSON to stand alone,
         extracted nothing. The model asked. The parser missed it. The loop
         reads silence as an answer.

    Case 3 is not hypothetical; it is measurable, and the next section is
    the story of provoking it by accident.

    This is why real agent frameworks do not infer completion from silence.
    They give the model an explicit `done` tool it must call, or verify the
    answer separately.

TRY IT WITH THE SMALL MODEL
    Change MODEL to "qwen2.5:1.5b-instruct" and run it again.

    In 06 that model made a correct single tool call 4 times out of 5 --
    basically as good as llama3.1. Here it calls list_files once, then
    stops using the tools and starts writing plausible-looking file
    contents from imagination. Two turns, zero files actually counted,
    confidently wrong.

    That contrast is the most useful thing in this file after the loop
    itself. "Supports tool calling" is a capability flag both models
    declare. Sustaining a multi-step loop -- keeping track of what it has
    learned, and continuing to use tools rather than inventing answers --
    is a separate and much harder capability that no flag reports.

THE BUDGET IS NOT DECORATION
    MAX_TURNS exists because models genuinely do get stuck: re-calling the
    same tool with the same arguments, or forgetting they already have what
    they need. Without a budget that is an infinite loop against a paid or
    slow endpoint. Watch for repeated calls in the output -- you will see it.

PREVIOUSLY
    06 made a single tool call. 07 wrapped repetition around a call without
    accumulating anything.

NEXT
    Nothing on the main line -- this is where the sequence lands. See
    examples/extras/ for optional directions, including giving it a voice.

RUN IT
    python examples/08_agent_loop.py            # or: make run-08
    python examples/08_agent_loop.py --gated    # or: make run-08-gated

    Run both. The default speculates on turn 1 and recovers; --gated makes
    speculation impossible: count_lines is not sent to the model at all
    until list_files has run.
    The contrast between the two traces is the most useful thing here.
"""

import sys
from pathlib import Path

from ollama import chat

MODEL = "llama3.1"
MAX_TURNS = 14

# Run with --gated to turn this on. It is the most interesting switch here:
#
#     python examples/08_agent_loop.py            # speculates, then recovers
#     python examples/08_agent_loop.py --gated    # cannot speculate at all
#
# When gated, count_lines is not advertised to the model until list_files has
# actually run. On turn 1 there is only one tool in existence, so the model
# CANNOT emit a speculative count_lines(file_name='script1.py') -- not because
# it was told not to, but because there is nothing to emit.
#
# Measured, 3 runs each:
#
#   GATE_TOOLS = False   9 calls on turn 1, 3/3 correct, 3 turns
#   GATE_TOOLS = True    1 call  on turn 1, 3/3 correct, 7 turns
#
# And the approach most people try first, asking politely, does not work:
# a system prompt saying "call EXACTLY ONE tool per reply, then STOP and wait"
# got 1 call on turn 1 but 3/3 PARSER_MISS -- the model narrates its plan in
# prose before the JSON, and Ollama's parser then extracts nothing.
#
# The general lesson is worth more than this example: constrain the INTERFACE,
# do not instruct the model. Removing an option produces obedience; asking for
# restraint produces prose.
GATE_TOOLS = False

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
    List every Python script available. Call this first -- you cannot know
    the filenames otherwise.
    """
    names = sorted(p.name for p in SCRIPTS_DIR.glob("*.py"))
    if not names:
        return "(no python files found)"

    # Tool output is prompt text too, and this sentence is load-bearing.
    #
    # Measured, 5 runs each, everything else identical:
    #
    #   returning "a.py\nb.py\nc.py"          -> 5/5 WRONG answers
    #   returning the sentence below          -> 5/5 CORRECT
    #
    # A bare newline list gets skimmed. Stating the count, enumerating the
    # names inline, and restating the obligation does not. Same lesson as
    # the docstring in 06, pointing the other way: what a tool RETURNS is
    # read by a language model, so write it for one.
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
    have_listed = False  # only used when GATE_TOOLS is on

    for turn in range(1, MAX_TURNS + 1):
        # Progressive disclosure. Note what this does NOT do: it does not
        # filter, suppress, or discard anything. It changes what is SENT.
        #
        # Gated turn 1 calls chat(..., tools=[list_files]). count_lines is
        # simply not in that list, so Ollama never renders its schema and the
        # string "count_lines" appears nowhere in the prompt. The model does
        # not know it exists.
        #
        # That is why this works where a system prompt does not. "Do not call
        # count_lines yet" still tells the model count_lines exists. This does
        # not mention it. A tool the model cannot see is a mistake it cannot
        # make.
        if GATE_TOOLS and not have_listed:
            available = [list_files]
        else:
            available = list(TOOLS.values())

        response = chat(
            model=MODEL,
            messages=messages,
            tools=available,
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

            if name == "list_files":
                have_listed = True  # unlocks count_lines when GATE_TOOLS is on

            # The model controls the tool NAME and the ARGUMENTS, and both are
            # just tokens it predicted. Neither can be trusted.
            #
            # An earlier version of this line was `tool(**args)` with no guard,
            # and it crashed on a real run: the model called
            # list_files(file_name='script1.py') -- an argument to a function
            # that takes none. TypeError, traceback, dead loop.
            #
            # Returning the error as a tool result instead is strictly better.
            # The model reads it and corrects itself, exactly as it does with
            # "File not found". A crash ends the run; a message continues it.
            #
            # And the WORDING of that message decides whether it recovers.
            # Measured, 5 runs each, only the error string differing:
            #
            #   "Bad arguments: {error}"                        -> 0/5 correct
            #   "Bad arguments for {name}: {error}. Check the
            #    tool's parameters and call it again."          -> 5/5 correct
            #
            # Without the closing instruction the model narrates its plan in
            # prose instead of emitting JSON, Ollama's parser finds no tool
            # call, and the loop exits early believing it got an answer --
            # case 3 in the header. Five words of instruction are the whole
            # difference between a loop that finishes and one that stalls.
            tool = TOOLS.get(name)

            if tool is None:
                result = (
                    f"Unknown tool: {name}. "
                    f"The available tools are: {', '.join(TOOLS)}."
                )
            else:
                try:
                    result = tool(**args)
                except TypeError as error:
                    result = (
                        f"Bad arguments for {name}: {error}. "
                        f"Check the tool's parameters and call it again."
                    )

            first_line = result.split("\n")[0]
            extra = f" (+{len(result.splitlines()) - 1} more lines)" if "\n" in result else ""
            print(f"          -> {first_line}{extra}")

            messages.append({"role": "tool", "content": result})

        print(f"          [context now {len(messages)} messages]")

    print(f"\n[gave up] hit the {MAX_TURNS}-turn budget without an answer.")
    print("This is the failure a budget exists to bound. Without it, this")
    print("loop would keep going until you killed it.")


if __name__ == "__main__":
    GATE_TOOLS = "--gated" in sys.argv

    if GATE_TOOLS:
        print("[mode] GATED -- turn 1 sends tools=[list_files] only.")
        print("       count_lines is not passed to the model at all, so it does")
        print("       not appear in the prompt and the model has no idea it")
        print("       exists. Nothing is suppressed; the option is absent.\n")

    else:
        print("[mode] UNGATED -- both tools offered from the start.")
        print("       Expect turn 1 to batch several calls with invented")
        print("       filenames. Re-run with --gated to make that impossible.\n")

    agent_loop(QUESTION)
