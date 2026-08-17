"""
02 -- Swap models and system prompts without editing code.

MAIN POINT
    Most of what feels like "the model's personality" is the system prompt,
    not the model. Holding one fixed while varying the other is the single
    most useful habit for learning what a model actually contributes.

    Same prompt on llama3.1 vs qwen2.5 tells you about the models.
    Same model with logic-bot vs chatty-bot tells you about prompting.

    Keep prompts in files and read the model list at runtime, and trying any
    combination costs a few keystrokes instead of an edit.

A NOTE ON WORDING
    A model is not a string. It is a few gigabytes of weights sitting on
    your disk. The string is how you ask Ollama to load one -- a handle, not
    the thing itself. Worth being precise about, because "the model is just
    a name" quietly suggests swapping them is cosmetic. It is not.

PREVIOUSLY
    01 showed that you maintain the message list yourself. Here the system
    prompt at the head of that list becomes something you choose.

NEXT
    03 adds the third control: how the model samples.

RUN IT
    python examples/02_pick_model_and_prompt.py
"""

from pathlib import Path
import sys

import ollama

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


##############################################################################
# COMMANDS                                                       [new in 02]
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
# FINDING WHAT IS AVAILABLE                                      [new in 02]
##############################################################################
# Ollama reports installed models. The field name has moved between versions,
# so all three spellings are checked -- the kind of small defensive detail
# that is invisible until it breaks.

def installed_models() -> list[str]:
    response = ollama.list()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])

    names = []
    for model in models or []:
        name = getattr(model, "model", None)
        if not name and isinstance(model, dict):
            name = model.get("model") or model.get("name")
        if name:
            names.append(str(name))

    return sorted(names)


def choose(label: str, choices: list[str]) -> str:
    if not choices:
        raise ValueError(f"Nothing available for {label.lower()}.")

    print(f"\n{label}")
    for number, choice in enumerate(choices, start=1):
        print(f"  {number:>2}. {choice}")

    while True:
        try:
            selection = input("\nSelection: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)

        try:
            index = int(selection) - 1
        except ValueError:
            print("Enter the number of your selection.")
            continue

        if 0 <= index < len(choices):
            return choices[index]

        print(f"Enter a number from 1 to {len(choices)}.")


##############################################################################
# THE CHAT LOOP                                                  [new in 02]
##############################################################################
# Same structure as 01 -- append, send everything, append the reply -- with
# streaming so the first words appear immediately, and the history left to
# grow unchecked. 04 is where that becomes a problem worth solving.

def run_chat(model: str, system_prompt_text: str) -> None:
    system_prompt = {"role": "system", "content": system_prompt_text}
    history: list[dict] = []

    print(f"\nModel: {model}")
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

        print("\nAssistant: ", end="", flush=True)
        parts = []

        try:
            for chunk in ollama.chat(
                model=model, messages=[system_prompt] + history, stream=True
            ):
                content = chunk.message.content or ""
                parts.append(content)
                print(content, end="", flush=True)
        except ollama.ResponseError as error:
            history.pop()  # drop the user turn so history stays consistent
            print(f"\n\nOllama error: {error.error}\n")
            continue

        history.append({"role": "assistant", "content": "".join(parts)})
        print(f"\n\n[history: {len(history)} messages]\n")


def main() -> int:
    try:
        model = choose("Models", installed_models())

        prompt_files = sorted(PROMPTS_DIR.glob("*.txt"))
        if not prompt_files:
            raise FileNotFoundError(f"No prompts found in {PROMPTS_DIR}")

        selected = choose("System prompts", [p.stem for p in prompt_files])
        system_prompt = (PROMPTS_DIR / f"{selected}.txt").read_text("utf-8").strip()

        if not system_prompt:
            raise ValueError(f"System prompt is empty: {selected}.txt")

        run_chat(model, system_prompt)
        return 0

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ConnectionError:
        print("Could not connect to Ollama. Is `ollama serve` running?", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
