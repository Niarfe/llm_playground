"""
09 -- A loop whose PATH changes. (Experimental.)

WHY THIS EXISTS
    08 demonstrates the agent loop mechanism honestly, but its task does not
    justify one. "List the files, count each, report the largest" has a
    sequence you can write down in advance -- so you could just write it
    down. It is a for-loop wearing a costume.

    Three shapes of problem, and only the third needs an agent:

      SUDOKU SHAPE       Feedback carries no information. Each attempt is an
                         independent draw with near-zero success. That is a
                         retry loop (07), and a bad one.

      PROCEDURE SHAPE    You can specify the steps. Then specify them --
                         `max(files, key=count_lines)` is three lines, needs
                         no model, and cannot hallucinate. This is 08's task,
                         and it is the failure mode hardest to notice, because
                         the loop appears to work.

      BRANCHING SHAPE    What you learn changes WHICH action comes next, not
                         merely the arguments to an action you had already
                         chosen. You cannot write the sequence down, because
                         step two depends on what step one found.

MAIN POINT
    An agent loop is warranted when the next action depends on information
    you can only obtain by acting -- AND that information changes which
    action you take, rather than filling in a blank.

    That second clause is the one everybody misses, this file's author
    included. 08 satisfies the first and fails the second: the filenames
    were unknown, but they only ever became arguments to count_lines, which
    was always going to be the next call. The plan never forked.

THE DEMONSTRATION
    Three cases. Same code, same tools, same prompt shape. The only thing
    that differs is the opening line of the case.

      case 1  a bullet wound   -> who could reach the locked gun cabinet
      case 2  a loose pearl    -> who owns pearls, and who was even here
      case 3  a locked cellar  -> who holds a key, and when the lock changed

    Run all three and watch the file sequence differ. That is the whole
    point, and it is the one thing 08 structurally cannot show you: its path
    is identical every run because its path was decided before it ran.

TWO DESIGN CHOICES WORTH COPYING
    The filenames do not give the game away. There is no `who_has_a_gun.txt`.
    Reaching for `household_inventory.txt` on a bullet wound is a small act
    of reasoning; reaching for a file whose name repeats the clue would only
    demonstrate string matching.

    No single file solves a case. The inventory names three keyholders; the
    alibis eliminate two. The register names two pearl owners; the guest book
    shows one was in London. The second file is chosen BECAUSE of what the
    first one said -- which is the fork this example exists to show.

ACCUSING IS AN EXPLICIT ACT
    08 stops when tool_calls comes back empty, and we established that means
    at least three different things: it answered, it gave up, or it asked for
    a tool and Ollama's parser missed it. Silence is a terrible completion
    signal.

    Here the loop ends only when the model calls accuse(). A detective who
    must formally name someone cannot finish by trailing off. This is what
    real agent frameworks do, and the fiction happens to supply the reason.

RUN IT
    python examples/09_branching_agent.py            # case 1
    python examples/09_branching_agent.py --case 2
    python examples/09_branching_agent.py --case 3

STATUS -- EXPERIMENTAL, AND HONESTLY REPORTED
    Not on the main line. First trial results, one run per case:

        qwen2.5:7b-instruct   case 1 CORRECT   case 2 CORRECT   case 3 WRONG
        llama3.1              case 1 CORRECT   case 2 WRONG     case 3 WRONG

    The idea works and the task is right. The reliability is not there yet.

    WHAT THE FAILURES ARE
    Both models have a strong prior toward opening alibis.txt regardless of
    the clue -- it is what a detective "should" read, so they read it. On
    case 3 that prior wins outright: the clue is a locked cellar DOOR, the
    relevant record is staff_records ("which doors each may unlock"), and
    the model goes to household_inventory and alibis anyway, then accuses
    whoever has the weakest alibi. It gets a plausible answer by the wrong
    route, which is the most dangerous kind of wrong.

    Note case 2 succeeded via a DIFFERENT second record than intended --
    alibis rather than the guest book -- and still reached Verity. Right
    answer, unplanned path. Worth knowing before trusting any single run.

    WHAT ALREADY WORKS AND IS WORTH KEEPING
    - The branch is real. On the runs that succeed, case 1 and case 2 open
      genuinely different first records. 08 cannot do that at all.
    - The accuse() preconditions earn their place immediately. The very
      first trial accused the VICTIM with an invented justification; the
      second accused "Blackwood", which is the house. Both are now refused
      with an explanation the model reads and recovers from.
    - The record index was necessary, not a nicety. Without it the task is
      circular: you cannot know which record is relevant without reading it.

    WHAT TO TRY NEXT
    - Break the alibis-first reflex. Perhaps alibis is only unlocked after
      a clue-specific record has been read (gating, as in 08).
    - Score the PATH, not just the verdict. Right answer by wrong route
      should not count as a pass.
    - Several runs per case before believing any number here.
"""

import sys
from pathlib import Path

from ollama import chat

MODEL = "llama3.1"
MAX_TURNS = 14

CASEFILES = Path(__file__).resolve().parent / "casefiles"


##############################################################################
# THE CASES                                                      [new in 09]
##############################################################################
# Each opens with a different physical clue, and each is solved by a
# different pair of files. Nothing else changes between them -- same tools,
# same system prompt, same loop.

CASES = {
    1: {
        "brief": (
            "Mr Havisham was found in the study of Blackwood Hall, killed by a "
            "single shot. The study display cabinet was locked and undamaged. "
            "Name the person responsible."
        ),
        "culprit": "Ashby",
        "expected_path": ["household_inventory.txt", "alibis.txt"],
    },
    2: {
        "brief": (
            "Mrs Fairlow was found in the conservatory of Blackwood Hall. A "
            "single loose pearl was in her closed hand. Name the person "
            "responsible."
        ),
        "culprit": "Verity",
        "expected_path": ["jewellery_register.txt", "guest_book.txt"],
    },
    3: {
        "brief": (
            "The wine cellar at Blackwood Hall was entered overnight and the "
            "door was found still locked, with no sign of forcing. Name the "
            "person responsible."
        ),
        "culprit": "Beel",
        "expected_path": ["staff_records.txt", "maintenance_log.txt"],
    },
}

SYSTEM_PROMPT = (
    "You are a careful detective. Work only from the case files. "
    "Call list_files first to see what records exist, then read only the "
    "records that bear on the clue you were given -- do not read all of them. "
    "A single record is rarely enough: if one narrows the field to several "
    "people, read a second record that can eliminate some of them. "
    "When, and only when, one person remains, call accuse with their surname "
    "and your reasoning. Never accuse without having read the records."
)


##############################################################################
# THE TOOLS                                                      [new in 09]
##############################################################################
# Same boundary discipline as 06 and 08: resolve, then check containment.
# The model chooses these filenames, so they are untrusted input.

# A bare list of filenames is not enough to choose from, and the first trial
# proved it: the model could not know that "household_inventory" describes the
# contents of the locked cabinet without reading it first. That is circular.
# It read records at random and confabulated.
#
# Every real archive has an index. One line per record, saying what KIND of
# information it holds -- not what it says. "Who holds keys to locked
# furniture" is a category; "Ashby has a cabinet key" would be the answer.
# The reasoning step survives; the mind-reading does not.
RECORD_INDEX = {
    "household_inventory.txt": "objects and furniture in each room, and who "
                               "holds keys to locked cabinets and drawers",
    "jewellery_register.txt": "insured items of value and their owners",
    "staff_records.txt": "employees, their positions, and which doors of the "
                         "building each may unlock",
    "alibis.txt": "where each person was on the evening in question, and who "
                  "can confirm it",
    "guest_book.txt": "who visited the house on which days, and who was away",
    "maintenance_log.txt": "repairs and alterations to the building, including "
                           "locks and keys",
    "library_catalogue.txt": "books held in the library",
    "menu_cards.txt": "meals served on each day",
}


def list_files() -> str:
    """
    List the case records available to you, with a note on what each one
    contains. Call this first -- you cannot know what records exist otherwise.
    """
    names = sorted(p.name for p in CASEFILES.glob("*.txt"))
    lines = [f"There are {len(names)} records:"]
    for name in names:
        lines.append(f"  {name} -- {RECORD_INDEX.get(name, 'uncategorised')}")
    lines.append(
        "Read only the records that bear on your clue. Reading all of them is "
        "not investigating."
    )
    return "\n".join(lines)


def read_file(file_name: str) -> str:
    """
    Read one case record.

    Args:
      file_name: Bare filename only, e.g. "alibis.txt". Must be a name that
        list_files returned. Do not include any directory path.
    """
    path = (CASEFILES / file_name).resolve()

    if not str(path).startswith(str(CASEFILES)) or path.suffix != ".txt":
        return f"Refused: {file_name} is not a case record."
    if not path.exists():
        return (
            f"No such record: {file_name}. Call list_files to see the real "
            f"names and try again."
        )

    return path.read_text().strip()


##############################################################################
# THE COMPLETION TOOL, WITH PRECONDITIONS                        [new in 09]
##############################################################################
# accuse() is the completion signal -- the loop ends when it succeeds, rather
# than when tool_calls comes back empty (which, per 08, means three different
# things and none of them reliably "finished").
#
# It also GUARDS. The first trial of this example failed exactly here: the
# model batched list_files + read_file + accuse into a single reply, read one
# record, and accused the victim, inventing a justification the file did not
# contain.
#
# Two preconditions fix that, and both are honest -- they encode what the task
# actually requires rather than leaking the answer:
#
#   1. At least two records read. One record never eliminates anyone here.
#   2. The accused must appear in a record you have actually read.
#
# A refusal comes back as a readable tool result, so the model can recover --
# the same principle as "File not found" in 08. A precondition that crashes
# teaches nothing; one that explains gets obeyed.

INVESTIGATION = {"records_read": {}}  # filename -> text, reset per case


def accuse(person: str, reasoning: str) -> str:
    """
    Formally accuse one person and close the case. Call this only once, after
    the records have narrowed the field to a single individual.

    Args:
      person: The surname of the person responsible.
      reasoning: Which records led you there, and what they ruled out.
    """
    read = INVESTIGATION["records_read"]

    if len(read) < 2:
        return (
            f"REFUSED. You have read {len(read)} record(s). One record never "
            f"narrows this to a single person -- it will name several. Read "
            f"another record that can eliminate some of them, then accuse."
        )

    surname = person.strip().split()[-1].lower() if person.strip() else ""

    # The brief names the victim and the house. Neither can be the culprit,
    # and a substring check alone lets both through -- an early run accused
    # "Blackwood", which is the building. Rejecting anything named in the
    # brief is principled: it rules out the victim and the setting without
    # revealing who the suspects are.
    if surname and surname in INVESTIGATION.get("brief", "").lower():
        return (
            f"REFUSED. '{person}' is named in the case brief -- that is the "
            f"victim or the location, not a suspect. Accuse someone the "
            f"records name."
        )

    mentioned_in = [f for f, text in read.items() if surname and surname in text.lower()]

    if not mentioned_in:
        return (
            f"REFUSED. '{person}' does not appear in any record you have read "
            f"({', '.join(read)}). Accuse only someone the records name. If "
            f"you meant the victim, note that the victim is not the culprit."
        )

    return f"ACCUSED: {person} -- case closed."


TOOLS = {"list_files": list_files, "read_file": read_file, "accuse": accuse}


##############################################################################
# THE LOOP                                                       [new in 09]
##############################################################################
# Structurally identical to 08. The difference is entirely in the task: here
# the second read_file is chosen because of what the first one returned.

def solve(case_number: int) -> dict:
    case = CASES[case_number]
    INVESTIGATION["records_read"] = {}
    INVESTIGATION["brief"] = case["brief"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case["brief"]},
    ]

    print(f"[case {case_number}] {case['brief']}\n")

    files_read: list[str] = []

    for turn in range(1, MAX_TURNS + 1):
        response = chat(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.values()),
            options={"temperature": 0},
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            # Not treated as an answer -- see the header. Nudge and continue.
            print(f"[turn {turn:>2}] (no tool call; prompting for a decision)")
            messages.append(
                {
                    "role": "user",
                    "content": "Continue. Read another record, or call accuse "
                    "if one person remains.",
                }
            )
            continue

        for call in response.message.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)
            print(f"[turn {turn:>2}] {name}({args})")

            tool = TOOLS.get(name)
            if tool is None:
                result = f"Unknown tool: {name}. Available: {', '.join(TOOLS)}."
            else:
                try:
                    result = tool(**args)
                except TypeError as error:
                    result = (
                        f"Bad arguments for {name}: {error}. Check the tool's "
                        f"parameters and call it again."
                    )

            if name == "read_file" and result.startswith(("Refused", "No such")):
                print(f"          -> {result.splitlines()[0]}")
            elif name == "read_file":
                fname = args.get("file_name", "?")
                files_read.append(fname)
                INVESTIGATION["records_read"][fname] = result
                print(f"          -> read {len(result.splitlines())} lines")
            else:
                print(f"          -> {result.splitlines()[0][:90]}")

            messages.append({"role": "tool", "content": result})

            if name == "accuse" and result.startswith("REFUSED"):
                print(f"          -> {result[:100]}")

            if name == "accuse" and not result.startswith("REFUSED"):
                print(f"\n[verdict] {args.get('person')}")
                print(f"[because] {args.get('reasoning', '')[:400]}")
                return {
                    "accused": args.get("person", ""),
                    "files_read": files_read,
                    "turns": turn,
                }

    print(f"\n[gave up] {MAX_TURNS} turns without an accusation.")
    return {"accused": "", "files_read": files_read, "turns": MAX_TURNS}


if __name__ == "__main__":
    number = 1
    if "--case" in sys.argv:
        number = int(sys.argv[sys.argv.index("--case") + 1])

    case = CASES[number]
    outcome = solve(number)

    correct = case["culprit"].lower() in outcome["accused"].lower()
    print(f"\n[expected]  {case['culprit']}  via {case['expected_path']}")
    print(f"[actual]    {outcome['accused'] or '(none)'}  via {outcome['files_read']}")
    print(f"[{'CORRECT' if correct else 'WRONG'}] in {outcome['turns']} turns")
    print(
        "\nRun the other cases. The tools and the loop do not change; the "
        "records\nit chooses to open do. That is the difference between a "
        "loop and a for-loop.\n"
    )
