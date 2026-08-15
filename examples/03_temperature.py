"""
03 — Temperature: the third thing you control.

MAIN POINT
    You control three things: which model, what prompt, and how it samples.
    02 covered the first two. Sampling is the third, and it is the one
    people leave at the default without realizing they made a choice.

    Temperature 0 makes the model deterministic — same input, same output,
    every time. Higher values make it sample more freely. Neither is
    "better"; they are for different jobs.

THE RULE WORTH REMEMBERING
    Steps that REPORT or EXTRACT want temperature 0.
    Steps that GENERATE want more.

    A step that reads a tool result and tells you what it said is a
    reporting step. Sampling there does not make it more creative, it makes
    it less reliable — it will hedge, embroider, or claim it is guessing at
    something it was handed. That is a real bug this repo hit in 06.

TEMPERATURE IS PER CALL, NOT PER APP
    One program can and should use different values at different steps.
    A Modelfile `PARAMETER temperature` sets a default baked into the
    model; `options={"temperature": N}` overrides it for a single call.

NEXT
    04 starts making decisions about what to send, not just how to sample it.

RUN IT
    python examples/03_temperature.py
"""

import ollama

MODEL = "llama3.1"

FACTUAL = "In one sentence, what does the Python len() function do?"
CREATIVE = "Write one sentence describing a thunderstorm."


##############################################################################
# TALKING TO OLLAMA                                              [new in 03]
##############################################################################
# `options` carries the sampling parameters. Everything not passed falls back
# to the model's own defaults, which is what "leaving it at the default"
# actually means.

def ask(prompt: str, temperature: float) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response.message.content.strip().replace("\n", " ")


def sample(label: str, prompt: str, temperature: float, n: int = 3) -> None:
    print(f"\n{label}  (temperature={temperature})")
    print(f"  prompt: {prompt}")
    outputs = [ask(prompt, temperature) for _ in range(n)]

    for i, out in enumerate(outputs, start=1):
        print(f"  {i}. {out[:110]}")

    identical = len(set(outputs)) == 1
    print(f"  -> {'IDENTICAL every run' if identical else f'{len(set(outputs))} different answers'}")


##############################################################################
# THE DEMONSTRATION                                              [new in 03]
##############################################################################
# Same prompt, same model, three samples each. The only thing changing is
# temperature. Run the file twice — the temperature 0 blocks will match
# across runs too, not just within one.

if __name__ == "__main__":
    print("=" * 74)
    print("A FACTUAL QUESTION — you want the same correct answer every time")
    print("=" * 74)
    sample("temperature 0", FACTUAL, 0.0)
    sample("temperature 1.2", FACTUAL, 1.2)

    print("\n" + "=" * 74)
    print("A CREATIVE REQUEST — sameness is now a defect, not a feature")
    print("=" * 74)
    sample("temperature 0", CREATIVE, 0.0)
    sample("temperature 1.2", CREATIVE, 1.2)

    print(
        "\nNotice temperature 0 repeats itself in BOTH cases. That is a win for\n"
        "the factual question and a failure for the creative one. Same setting,\n"
        "opposite verdict — which is why it belongs on the call, not the app.\n"
    )
