"""
07 — Let the model call a function on your machine.

WHAT THIS TEACHES
    Tool calling, which is the whole basis of "agents". The mechanic is
    much dumber than it sounds:

      1. You describe some functions to the model.
      2. Instead of prose, it replies with "call execute_local_script
         with script_path='./hello.py'". It cannot run anything -- it
         emits a request.
      3. YOUR code runs the function. This is the step people skip when
         they imagine agents. The model never touches your machine.
      4. You append the result as a "tool" message and call again.
      5. Now it answers using the result.

    Everything marketed as an agent is this loop, plus a while-loop, plus
    error handling.

SECURITY, PLAINLY
    This example hands the model a function that executes Python files.
    That is fine here because the model can only name a path and the only
    script present is hello.py. It would NOT be fine with a tool that ran
    arbitrary strings as shell commands -- at that point anything that can
    influence the conversation can run code as you.

    The rule: the tool defines the boundary, not the prompt. Do not write
    a permissive tool and rely on asking the model nicely.

MODEL SUPPORT
    Not every model does tool calling. llama3.1 and qwen2.5 do; some
    others silently answer in prose instead. If you get no tool_calls
    back, that is usually the reason rather than a bug in your code.

RUN IT
    ollama pull llama3.1
    python examples/07_tool_calling.py
"""

import subprocess
import sys
from pathlib import Path

from ollama import chat

MODEL = "llama3.1"
HERE = Path(__file__).resolve().parent


def execute_local_script(script_name: str) -> str:
    """
    Execute a Python script from the examples directory and return its output.

    Args:
      script_name: Bare filename only, e.g. "hello.py". Do not include any
        directory path. Do not invent a placeholder path.
    """
    # That docstring is not documentation -- it is the description the model
    # sees, and it was written by testing. The first version said only
    # "Execute a local Python script", and both llama3.1 and qwen2.5 called
    # it with "/path/to/hello.py" -- a placeholder, which the boundary check
    # below then refused. Adding the two constraint sentences took it to 6/6
    # correct across both models. Docstrings on tools are prompt engineering.
    path = (HERE / script_name).resolve()

    # Keep the model inside this directory. A tool with no boundary is a
    # tool that will eventually be pointed somewhere you did not intend.
    if not str(path).startswith(str(HERE)):
        return f"Refused: {script_name} is outside the examples directory."
    if not path.exists():
        return f"File not found: {script_name}"

    result = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=30
    )
    return result.stdout or result.stderr or "(no output)"


def main() -> int:
    messages = [
        {"role": "user", "content": "Run the script hello.py and tell me what it printed."}
    ]

    # The ollama SDK reads the signature and docstring to build the schema
    # the model sees. Nothing is registered anywhere -- passing the function
    # object is the whole setup.
    response = chat(model=MODEL, messages=messages, tools=[execute_local_script])

    if not response.message.tool_calls:
        print("The model did not request a tool. It said:")
        print(response.message.content)
        print(f"\n(Does {MODEL} support tool calling? Try llama3.1 or qwen2.5.)")
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

    final = chat(model=MODEL, messages=messages)
    print(f"\n[model's answer] {final.message.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
