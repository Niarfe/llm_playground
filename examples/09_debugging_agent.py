"""
09 -- Fix a failing test. This is what agents are actually for.

WHY THIS ONE
    08 demonstrates the loop mechanism, but its task does not justify one:
    "list the files, count each, report the largest" is a sequence you can
    write down in advance, so you could just write it down. A for-loop in
    costume.

    Three shapes of problem, and only the third needs an agent:

      SUDOKU SHAPE     Feedback carries no information. Every attempt is an
                       independent draw with near-zero success. Retry loop
                       (07), and a bad one.

      PROCEDURE SHAPE  You can specify the steps -- so specify them.
                       `max(files, key=count_lines)` is three lines, needs
                       no model, and cannot hallucinate. This is 08.

      BRANCHING SHAPE  What you learn changes WHICH action comes next, not
                       just the arguments to an action you had already
                       chosen. You cannot write the sequence down, because
                       step two depends on what step one found.

    Debugging is the canonical branching problem, and it is not a contrived
    one. It is the single most successful thing agents do commercially --
    Claude Code, Cursor and the rest are running this loop.

MAIN POINT
    An agent loop is warranted when the next action depends on information
    you can only get by acting -- AND that information changes which action
    you take.

    Here the test output decides everything. An assertion about
    rectangle_area sends you to shapes.py; one about reverse_words sends you
    to strings.py. You cannot know which before running the tests, and no
    amount of planning gets you there.

THE VERIFIER IS THE POINT
    The whodunit version of this example needed an answer key and a scoring
    script -- somebody had to decide what "right" meant.

    Here the test suite decides. It passes or it does not. Nothing in this
    file knows which line is wrong, there is no solutions.json, and the
    agent cannot talk its way to success.

    That is not a convenience, it is the reason debugging agents work better
    than research agents. Same loop, but one has cheap honest ground truth
    every iteration and the other does not. When you are judging whether
    some agentic idea will work, ask what its test suite is.

THE AGENT WRITES CODE THAT WE THEN RUN
    Which is exactly what real coding agents do, and exactly why they run in
    sandboxes. Mitigations here:

      - It works on a COPY in runs/workspace/, never the repo. Wreck it and
        the next run starts clean.
      - replace_in_file only touches .py files inside that workspace,
        and refuses any edit that would not compile.
      - Tests run in a subprocess with a timeout.

    The tool is the boundary, same as 06. Do not write a permissive tool and
    then ask the model nicely.

THE DEMONSTRATION
    Three scenarios. Identical code, identical tools, identical prompt. Only
    the injected bug differs:

      scenario 1  rectangle_area returns w + h   -> shapes.py
      scenario 2  reverse_words reverses letters -> strings.py
      scenario 3  average divides by len + 1     -> numbers.py

    Run all three and watch which file it opens. That is the branch, and 08
    structurally cannot show it -- 08's path is identical every run because
    its path was decided before it ran.

WHAT IT TOOK TO MAKE THIS WORK
    Four failures, each fixed by a technique this repo already established.
    Recorded because the fixes are the content:

    1. run_tests returned the raw traceback -- absolute paths, caret markers,
       forty lines around one useful sentence. The model replied in PROSE,
       invented a file, and wrote its fix in a markdown fence. Twelve turns,
       nothing done. Fix: return a signal, not a dump. (Tool output is prompt
       text -- the same lesson as 06's docstring and 08's file listing.)

    2. The signal named no function. The extractor matched the first line
       containing "AssertionError", which was the SOURCE line in the
       traceback frame, not the message. So the model chose a file at random
       and read shapes.py for all three bugs. It looked like a model reflex.
       It was my bug, and the "reflex" was the model having nothing to go on.

    3. The edit tool took the complete new file contents. llama3.1 cannot
       photocopy: instead of reproducing forty lines with one changed, it
       wrote a fresh file from imagination and then iterated on its own
       invention. Fix: replace_in_file, two short strings, cannot destroy
       what it does not mention. This is why real coding agents expose an
       edit-by-replacement tool.

    4. Turn 1 batched everything before seeing any output -- run_tests plus a
       read of "geometry.py" plus an edit to it, all composed blind. The
       invented name then poisoned every later turn. Fix: gate the tools, as
       in 08. Until the tests have run, run_tests is the only tool offered.

    Measured after all four, llama3.1, one run per scenario:
    3/3 PASS in 5, 5 and 7 turns, each editing a different file.

PREVIOUSLY
    08 built the loop. This gives it a job that needs one.

RUN IT
    make run-09            # scenario 1
    make run-09-all        # all three

    Or: python examples/09_debugging_agent.py --scenario 2
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

from ollama import chat

MODEL = "llama3.1"
MAX_TURNS = 12

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "broken"
WORKSPACE = HERE.parent / "runs" / "workspace"

SYSTEM_PROMPT = (
    "You are a careful engineer fixing a bug. Always call run_tests FIRST to "
    "see what is failing. The failure message names a function -- use it to "
    "decide which file to read. Read that file before changing anything. "
    "Fix it with replace_in_file: copy the exact wrong line as old_text and "
    "give the corrected line as new_text. Change one line if you can. "
    "After editing, call run_tests again to confirm. Stop when they pass."
)


##############################################################################
# THE WORKSPACE                                                  [new in 09]
##############################################################################
# The agent edits files. It gets a scratch copy so a bad edit cannot damage
# the repo, and so every run starts from the same broken state.

def prepare_workspace(scenario: int) -> Path:
    source = SCENARIOS / f"scenario{scenario}"
    if not source.exists():
        raise SystemExit(f"No such scenario: {source}")

    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)

    for path in source.glob("*.py"):
        shutil.copy(path, WORKSPACE / path.name)

    return WORKSPACE


##############################################################################
# THE TOOLS                                                      [new in 09]
##############################################################################
# run_tests is the verifier. Note that it reports failure as ordinary text,
# not an exception -- the model reads it and decides what to do, exactly as
# it reads "File not found" in 08.

def run_tests() -> str:
    """
    Run the test suite and return the result. Call this first to find out
    what is broken, and again after every edit to check your fix.
    """
    result = subprocess.run(
        [sys.executable, "test_all.py"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=30,
    )
    raw = (result.stdout + result.stderr).strip()

    if "ALL TESTS PASSED" in raw:
        return "ALL TESTS PASSED. The bug is fixed. Stop here."

    # Return a SIGNAL, not a dump.
    #
    # The first version returned the raw traceback: absolute paths, Python's
    # ~~~^^^ caret markers, forty lines of frame noise around one useful
    # sentence. The model responded with prose -- it narrated a plan, invented
    # a filename that did not exist, and wrote the fix in a markdown fence
    # instead of calling a tool at all. The loop stalled for twelve turns.
    #
    # Same lesson as the tool docstring in 06 and the file listing in 08, in
    # its third costume: what a tool RETURNS is read by a language model, so
    # write it for one. A traceback is written for a human with a debugger.
    # Take the LAST line matching "SomeError: message". The first version
    # searched for any line containing "AssertionError", which matched the
    # source line shown in the traceback frame --
    #
    #   raise AssertionError(f"{label}: expected {expected!r}, ...")
    #
    # -- rather than the exception message itself. So the signal named no
    # function at all, and the model picked a file at random. It read
    # shapes.py for all three bugs and looked like a reflex problem. It was
    # a bug in the tool, and the "reflex" was the model having nothing to
    # go on.
    errors = [
        line.strip() for line in raw.splitlines()
        if re.match(r"^\s*\w*(Error|Exception):", line)
    ]
    failure = errors[-1] if errors else ""
    passed = [line.strip() for line in raw.splitlines() if line.strip().startswith("ok")]

    available = ", ".join(sorted(p.name for p in WORKSPACE.glob("*.py")))
    return (
        f"TESTS FAILED.\n"
        f"{len(passed)} checks passed before the failure.\n"
        f"Failure: {failure or raw.splitlines()[-1]}\n"
        f"The project files are: {available}\n"
        f"Decide which file defines the failing function, call read_file on "
        f"it, then call replace_in_file with the wrong line and its correction."
    )


def list_files() -> str:
    """List the Python files in the project you are fixing."""
    names = sorted(p.name for p in WORKSPACE.glob("*.py"))
    return f"The project has {len(names)} files: " + ", ".join(names)


def read_file(file_name: str) -> str:
    """
    Read one source file.

    Args:
      file_name: Bare filename only, e.g. "shapes.py". No directory path.
    """
    path = (WORKSPACE / file_name).resolve()
    if not str(path).startswith(str(WORKSPACE)) or path.suffix != ".py":
        return f"Refused: {file_name} is not a Python file in this project."
    if not path.exists():
        return f"No such file: {file_name}. Call list_files to see real names."
    return path.read_text()


# The first version of this tool took the COMPLETE new file contents, which
# is how you would naively design it. llama3.1 could not do it: rather than
# reproduce the file with one line changed, it wrote a fresh file from
# imagination -- `import math`, a `square()` function that was never there --
# then spent the remaining turns iterating on its own invention. The original
# code was gone by turn 3.
#
# Asking an 8B model to reproduce forty lines verbatim to change one of them
# is asking it to be a photocopier, which it is not. A surgical replace asks
# for two short strings instead, and cannot destroy what it does not mention.
#
# This is why real coding agents expose an edit-by-replacement tool rather
# than a write-whole-file tool. Tool granularity is a design decision with
# consequences, not a detail.
def replace_in_file(file_name: str, old_text: str, new_text: str) -> str:
    """
    Replace one exact snippet of a file with new text. Use this to fix a bug:
    give the single line you want to change and what it should become.

    Args:
      file_name: Bare filename only, e.g. "shapes.py". No directory path.
      old_text: The exact text to replace, copied from the file. Must appear
        exactly once. Usually one line.
      new_text: What that text should become.
    """
    path = (WORKSPACE / file_name).resolve()
    if not str(path).startswith(str(WORKSPACE)) or path.suffix != ".py":
        return f"Refused: {file_name} is not a Python file in this project."
    if not path.exists():
        return f"No such file: {file_name}. Call list_files to see real names."

    source = path.read_text()
    occurrences = source.count(old_text)

    if occurrences == 0:
        return (
            f"Not found: that exact text does not appear in {file_name}. "
            f"Call read_file and copy the line exactly as it is written."
        )
    if occurrences > 1:
        return (
            f"Ambiguous: that text appears {occurrences} times in {file_name}. "
            f"Include more surrounding lines to make it unique."
        )

    updated = source.replace(old_text, new_text)

    try:
        compile(updated, file_name, "exec")
    except SyntaxError as error:
        return (
            f"Refused: that edit makes {file_name} invalid Python "
            f"({error.msg} on line {error.lineno}). The file was NOT changed."
        )

    path.write_text(updated)
    return f"Edited {file_name}. Now call run_tests."


TOOLS = {
    "run_tests": run_tests,
    "list_files": list_files,
    "read_file": read_file,
    "replace_in_file": replace_in_file,
}


##############################################################################
# THE LOOP                                                       [new in 09]
##############################################################################
# Same shape as 08. The difference is that success is decided by running the
# code, not by the model announcing it is finished.

def fix_it(scenario: int) -> dict:
    prepare_workspace(scenario)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "The test suite is failing. Find the bug and fix it."},
    ]
    print(f"[scenario {scenario}] workspace: {WORKSPACE}")
    print(f"[user] The test suite is failing. Find the bug and fix it.\n")

    files_read: list[str] = []
    files_written: list[str] = []

    have_run_tests = False

    for turn in range(1, MAX_TURNS + 1):
        # Gating, exactly as in 08 -- and here it is necessary rather than
        # optional. Until the tests have actually run, the ONLY tool offered
        # is run_tests.
        #
        # Ungated, llama3.1 batches its whole plan into turn 1: run_tests,
        # plus a read_file for "geometry.py" (a file that does not exist),
        # plus an edit to it. All composed before seeing a single line of
        # output. The invented filename then sits in the context poisoning
        # every later turn -- in one run it correctly worked out that the fix
        # was `sum(values) / len(values)` and spent four turns trying to apply
        # it to shapes.py.
        #
        # It cannot speculate about a file it has not been told exists, so
        # do not offer it the means until it has looked.
        available = [run_tests] if not have_run_tests else list(TOOLS.values())

        response = chat(
            model=MODEL,
            messages=messages,
            tools=available,
            options={"temperature": 0},
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            print(f"[turn {turn:>2}] (no tool call; nudging)")
            messages.append({
                "role": "user",
                "content": "Continue. Call run_tests to check the current state.",
            })
            continue

        for call in response.message.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)

            shown = {k: (v[:40] + "...") if isinstance(v, str) and len(v) > 40 else v
                     for k, v in args.items()}
            print(f"[turn {turn:>2}] {name}({shown})")

            if name == "run_tests":
                have_run_tests = True

            tool = TOOLS.get(name)
            if tool is None:
                result = f"Unknown tool: {name}. Available: {', '.join(TOOLS)}."
            else:
                try:
                    result = tool(**args)
                except TypeError as error:
                    result = (f"Bad arguments for {name}: {error}. Check the "
                              f"tool's parameters and call it again.")
                except subprocess.TimeoutExpired:
                    result = "The tests timed out. Did an edit introduce a loop?"

            # Only record a read that actually happened. A call with the
            # wrong parameter name (the model sometimes sends `filename`)
            # returns a Bad arguments message, and counting that as a read
            # put a bogus '?' in the reported path.
            if name == "read_file" and result.startswith(('"""', "#", "import", "from", "def")):
                files_read.append(args["file_name"])
            if name == "replace_in_file" and result.startswith("Edited"):
                files_written.append(args.get("file_name", "?"))

            first = result.splitlines()[0] if result.splitlines() else ""
            print(f"          -> {first[:88]}")

            messages.append({"role": "tool", "content": result})

            # Ground truth. Not the model's opinion, not a scoring script --
            # the code ran and this is what happened.
            if name == "run_tests" and "ALL TESTS PASSED" in result:
                print(f"\n[FIXED] tests pass after {turn} turns")
                return {"passed": True, "turns": turn,
                        "read": files_read, "written": files_written}

    print(f"\n[gave up] {MAX_TURNS} turns, tests still failing")
    return {"passed": False, "turns": MAX_TURNS,
            "read": files_read, "written": files_written}


if __name__ == "__main__":
    scenario = 1
    if "--scenario" in sys.argv:
        scenario = int(sys.argv[sys.argv.index("--scenario") + 1])
    if "--model" in sys.argv:
        globals()["MODEL"] = sys.argv[sys.argv.index("--model") + 1]

    print(f"[model] {MODEL}")
    outcome = fix_it(scenario)

    print(f"[read]    {outcome['read'] or '(nothing)'}")
    print(f"[wrote]   {outcome['written'] or '(nothing)'}")
    print(f"[result]  {'PASS' if outcome['passed'] else 'FAIL'} "
          f"in {outcome['turns']} turns\n")

    raise SystemExit(0 if outcome["passed"] else 1)
