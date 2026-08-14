from pathlib import Path
import sys
import pryzm as pz

import ollama

agent = pz.Pryzm(echo=True).yellow
user  = pz.Pryzm(echo=False).cyan

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"


def model_name(model: object) -> str:
    """Extract a model name from Ollama's response object."""
    name = getattr(model, "model", None)

    if name:
        return str(name)

    if isinstance(model, dict):
        return str(model.get("model") or model.get("name") or "")

    return ""


def get_installed_models() -> list[str]:
    """Return the locally available Ollama model names."""
    response = ollama.list()
    models = getattr(response, "models", None)

    if models is None and isinstance(response, dict):
        models = response.get("models", [])

    names = [model_name(model) for model in models or []]
    return sorted(name for name in names if name)


def get_prompt_files() -> list[Path]:
    """Return available system-prompt text files."""
    if not PROMPTS_DIR.exists():
        raise FileNotFoundError(
            f"Prompt directory does not exist: {PROMPTS_DIR}"
        )

    return sorted(PROMPTS_DIR.glob("*.txt"))


def choose(label: str, choices: list[str]) -> str:
    """Let the user choose one item from a numbered list."""
    if not choices:
        raise ValueError(f"No choices available for {label.lower()}.")

    print(f"\n{label}")

    for number, choice in enumerate(choices, start=1):
        print(f"  {number:>2}. {choice}")

    while True:
        selection = input("\nSelection: ").strip()

        try:
            index = int(selection) - 1
        except ValueError:
            print("Enter the number of your selection.")
            continue

        if 0 <= index < len(choices):
            return choices[index]

        print(f"Enter a number from 1 to {len(choices)}.")


def run_chat(model: str, system_prompt: str) -> None:
    """Run an interactive Ollama chat session."""
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    print(f"\nModel: {model}")
    print("Commands: /clear, /system, /exit\n")

    while True:
        try:
            user_input = input(user("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "/quit"}:
            return

        if user_input.lower() == "/clear":
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ]
            print("Conversation cleared.\n")
            continue

        if user_input.lower() == "/system":
            print(f"\n--- System prompt ---\n{system_prompt}\n")
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        #print("\nAssistant: ", end="", flush=True)
        agent("\nAssistant: ")
        response_parts: list[str] = []

        try:
            stream = ollama.chat(
                model=model,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                content = chunk.message.content or ""
                response_parts.append(content)
                print(content, end="", flush=True)

        except ollama.ResponseError as error:
            messages.pop()
            print(f"\n\nOllama error: {error.error}\n")
            continue

        assistant_response = "".join(response_parts)
        messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )

        print("\n")


def main() -> int:
    try:
        models = get_installed_models()
        prompt_files = get_prompt_files()

        model = choose("Models", models)

        prompt_labels = [path.stem for path in prompt_files]
        selected_prompt = choose("System prompts", prompt_labels)
        prompt_path = PROMPTS_DIR / f"{selected_prompt}.txt"

        system_prompt = prompt_path.read_text(encoding="utf-8").strip()

        if not system_prompt:
            raise ValueError(f"System prompt is empty: {prompt_path}")

        run_chat(model, system_prompt)
        return 0

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    except ConnectionError:
        print(
            "Could not connect to Ollama. Verify that Ollama is running.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
