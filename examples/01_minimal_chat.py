"""
01 — The smallest thing that works.

WHAT THIS TEACHES
    Two ways to get text out of a local Ollama model, and why you almost
    always want the second one.

    generate()  takes a single prompt string. No conversation, no memory.
    chat()      takes a list of messages. This is the shape every later
                example builds on.

WHY IT MATTERS
    The `messages` list is the whole ballgame. Every idea in this repo --
    system prompts, compaction, fact memory -- is just a different strategy
    for deciding what goes in that list before you send it.

RUN IT
    ollama serve          # in another terminal
    ollama pull dolphin3
    python examples/01_minimal_chat.py
"""

import ollama

MODEL = "dolphin3"

# --- One-shot: no history, no roles. Good for a smoke test, little else. ---

print("--- generate() ---")
response = ollama.generate(model=MODEL, prompt="Say hello in five words.")
print(response.response.strip())


# --- Chat: a list of messages, each with a role. ---
#
# Roles are "system" (instructions), "user" (you), "assistant" (the model).
# The model sees this entire list on every single call -- it has no memory
# of its own. "Conversation" is an illusion you maintain by resending the
# whole transcript each time. That is exactly why it grows without bound,
# which is the problem example 03 solves.

print("\n--- chat() ---")
messages = [
    {"role": "system", "content": "You are terse. Answer in one sentence."},
    {"role": "user", "content": "What is a large language model?"},
]

response = ollama.chat(model=MODEL, messages=messages)
print(response.message.content.strip())

# To continue the conversation you append the reply and send everything again:
messages.append({"role": "assistant", "content": response.message.content})
messages.append({"role": "user", "content": "Now explain it to a six year old."})

response = ollama.chat(model=MODEL, messages=messages)
print("\n" + response.message.content.strip())

print(f"\n[{len(messages) + 1} messages in history -- and it only grows from here]")
