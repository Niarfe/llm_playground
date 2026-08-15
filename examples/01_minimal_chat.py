"""
01 — The smallest thing that works.

MAIN POINT
    The model has no memory. "Conversation" is an illusion you maintain by
    resending the entire transcript on every single call.

    Once that clicks, most of this repo follows from it. Compaction, fact
    stores, agent loops — they are all strategies for deciding what goes in
    that list before you send it.

IF OLLAMA ALREADY REMEMBERS, WHY BOTHER?
    Reasonable objection: `ollama run llama3.1` in a terminal clearly
    remembers what you said three turns ago. So the model does have memory?

    No. The CLI is keeping a message list for you and resending it, exactly
    like this file does — it just hides it. That is fine until you want a
    say in what gets kept, what gets dropped when the window fills, or what
    else gets injected. Then you need the loop back, which is what every
    example after this one is about.

NEXT
    02 varies the model and the prompt. 04 deals with the fact that this
    list only ever grows.

RUN IT
    ollama serve          # in another terminal
    ollama pull llama3.1
    python examples/01_minimal_chat.py
"""

import ollama

MODEL = "llama3.1"


##############################################################################
# ONE SHOT — generate()                                          [new in 01]
##############################################################################
# A single prompt string in, a single completion out. No roles, no history.
# Useful as a smoke test that Ollama is up, and for almost nothing else.

print("--- generate() ---")
response = ollama.generate(model=MODEL, prompt="Say hello in five words.")
print(response.response.strip())


##############################################################################
# A CONVERSATION — chat()                                        [new in 01]
##############################################################################
# A list of messages, each tagged with a role:
#   system     standing instructions
#   user       you
#   assistant  the model
#
# The model receives this WHOLE list every call. It remembers nothing between
# calls. Watch the count at the bottom.

print("\n--- chat() ---")
messages = [
    {"role": "system", "content": "You are terse. Answer in one sentence."},
    {"role": "user", "content": "What is a large language model?"},
]

response = ollama.chat(model=MODEL, messages=messages)
print(f"[sent {len(messages)} messages]")
print(response.message.content.strip())


##############################################################################
# CONTINUING IT                                                  [new in 01]
##############################################################################
# To continue, append the reply and resend everything. There is no other
# mechanism. Nothing is stored on the Ollama side between these two calls.

messages.append({"role": "assistant", "content": response.message.content})
messages.append({"role": "user", "content": "Now explain it to a six year old."})

response = ollama.chat(model=MODEL, messages=messages)
print(f"\n[sent {len(messages)} messages]")
print(response.message.content.strip())

messages.append({"role": "assistant", "content": response.message.content})

print(f"\n[history: {len(messages)} messages / {sum(1 for m in messages if m['role'] == 'user')} user turns]")
print("Each exchange adds TWO messages — one yours, one the model's.")
print("That is why context fills about twice as fast as people expect. See 04.")
