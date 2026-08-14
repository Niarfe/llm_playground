"""
02 — Swap models and system prompts without editing code.

WHAT THIS TEACHES
    A system prompt is just a string, and a model is just a name. If you
    keep prompts in files and read the model list at runtime, you can try
    any combination of the two in a few keystrokes.

WHY IT MATTERS
    Most of what feels like "the model's personality" is the system prompt,
    not the model. Being able to hold one fixed while varying the other is
    the single most useful habit for learning what a model actually does.

    Compare: the same prompt on dolphin3 vs deepseek-r1 tells you about the
    models. The same model with logic-bot vs chatty-bot tells you about
    prompting.

RUN IT
    python examples/02_pick_model_and_prompt.py

    Commands during chat: /clear  /system  /exit
"""

from pathlib import Path
import sys

import ollama

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def installed_models() -> list[str]:
    """Ask Ollama what is available locally."""
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
    """Numbered menu. Loops until the input is a valid index."""
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


def run_chat(model: str, system_prompt: str) -> None:
    """Stream a conversation, keeping full history (no compaction -- see 03)."""
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\nModel: {model}")
    print("Commands: /clear, /system, /exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit"}:
            return
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("Conversation cleared.\n")
            continue
        if user_input.lower() == "/system":
            print(f"\n--- System prompt ---\n{system_prompt}\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        parts = []

        try:
            # stream=True yields chunks as the model produces them, so the
            # first words appear immediately instead of after the full reply.
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                content = chunk.message.content or ""
                parts.append(content)
                print(content, end="", flush=True)
        except ollama.ResponseError as error:
            messages.pop()  # drop the user turn so history stays consistent
            print(f"\n\nOllama error: {error.error}\n")
            continue

        messages.append({"role": "assistant", "content": "".join(parts)})
        print("\n")


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
