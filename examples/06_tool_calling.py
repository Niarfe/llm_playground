"""
06 — Let the model call a function on your machine.

MAIN POINT
    The model cannot run anything. It emits a *request* to run something,
    and your code decides whether to honour it. That is the entire
    mechanism, and everything marketed as an agent is built on it.

      1. You describe some functions to the model.
      2. It replies with "call execute_local_script(script_name='hello.py')"
         instead of prose.
      3. YOUR code runs the function. This is the step people skip when
         imagining agents. The model never touches your machine.
      4. You append the result as a "tool" message and call again.
      5. It answers using the result.

    One round trip. 08 turns it into a loop, which is where it gets useful.

SECURITY, PLAINLY
    The tool defines the boundary — not the prompt. This one only runs files
    inside examples/scripts/, checked after resolving the path so that
    "../04_compaction.py" is refused rather than launching a chat loop.

    Do not write a permissive tool and rely on asking the model nicely. The
    model is not the attacker; anything that can influence the conversation
    is.

MODEL SUPPORT IS NOT UNIVERSAL
    llama3.1 and qwen2.5:7b support tool calling. qwen2.5:1.5b mostly does
    not — it answers in prose instead. That is not a bug in your code, and
    it is a cheap way to see that capability tracks model size. Try it.

PREVIOUSLY
    01-05 shaped what the model receives. This is the first time it acts.

NEXT
    07 shows a loop that retries. 08 shows a loop that makes progress.

RUN IT
    python examples/06_tool_calling.py
"""

import subprocess
import sys
from pathlib import Path

from ollama import chat

MODEL = "llama3.1"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


##############################################################################
# THE TOOL                                                       [new in 06]
##############################################################################
# The docstring is not documentation -- it is the description the model sees,
# and it was written by testing. The first version said only "Execute a local
# Python script", and both llama3.1 and qwen2.5 called it with
# "/path/to/hello.py", a placeholder, which the boundary check then refused.
# Adding the two constraint sentences took it to 6/6 correct across both
# models. Docstrings on tools are prompt engineering.

def execute_local_script(script_name: str) -> str:
    """
    Execute a Python script from the scripts directory and return its output.

    Args:
      script_name: Bare filename only, e.g. "hello.py". Do not include any
        directory path. Do not invent a placeholder path.
    """
    path = (SCRIPTS_DIR / script_name).resolve()

    # Resolve FIRST, then check containment -- otherwise "../04_compaction.py"
    # slips through and launches an interactive chat loop.
    if not str(path).startswith(str(SCRIPTS_DIR)):
        return f"Refused: {script_name} is outside the scripts directory."
    if not path.exists():
        return f"File not found: {script_name}"

    result = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=30
    )
    return result.stdout or result.stderr or "(no output)"


##############################################################################
# ONE ROUND TRIP                                                 [new in 06]
##############################################################################

def main() -> int:
    prompt = "Run the script hello.py and tell me what it printed."
    messages = [{"role": "user", "content": prompt}]

    # Echo it. This example types on your behalf, and watching a conversation
    # where you cannot see half the turns is needlessly confusing. You could
    # type this yourself -- that is the point.
    print(f"[user] {prompt}")

    # Passing the function object IS the setup. The SDK reads its signature
    # and docstring to build the schema. Nothing is registered anywhere.
    response = chat(model=MODEL, messages=messages, tools=[execute_local_script])

    if not response.message.tool_calls:
        print("\nThe model did not request a tool. It said:")
        print(response.message.content)
        print(f"\n(Does {MODEL} support tool calling? Try llama3.1 or qwen2.5:7b.)")
        return 1

    messages.append(response.message)

    for call in response.message.tool_calls:
        print(f"[model requested] {call.function.name}({call.function.arguments})")

        if call.function.name != "execute_local_script":
            messages.append({"role": "tool", "content": "Unknown tool."})
            continue

        # THIS is where execution happens -- in your process, under your rules.
        output = execute_local_script(**call.function.arguments)
        print(f"[we ran it, got] {output.strip()}")
        messages.append({"role": "tool", "content": output})

    # Pass tools= AGAIN. It looks redundant -- we are not asking for another
    # call -- but omitting it breaks the answer: the model replies "you didn't
    # provide the script contents" as though the tool never ran. The schema is
    # part of the chat template, so without it the template has no rule for
    # rendering a "tool" message and the result is invisible. Tool definitions
    # belong on every call in the exchange.
    #
    # temperature 0 because this call REPORTS a fact already in `messages`.
    # At the default, llama3.1 sometimes narrates the tool output as a guess.
    final = chat(
        model=MODEL,
        messages=messages,
        tools=[execute_local_script],
        options={"temperature": 0},
    )
    print(f"\n[model's answer] {final.message.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
