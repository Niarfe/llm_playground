"""
Ollama chat loop with:
  1. Rolling compaction to keep live prompt small.
  2. Local fact store (facts.json) populated during compaction.
  3. Cheap keyword-based retrieval against fact store.
  4. macOS native Text-to-Speech (`say`) toggling via /voice or /say commands.

Requires: pip install requests
Assumes Ollama is running locally: `ollama serve`.
"""

import json
import re
import shlex
import subprocess
import sys
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
#MODEL = "hermes3:latest"  # Optimized target for 16GB Apple Silicon (or use llama3.1)
MODEL = "dolphin3:latest"  # Optimized target for 16GB Apple Silicon (or use llama3.1)

SUMMARIZE_EVERY = 6
KEEP_LAST = 4

FACTS_PATH = Path("facts.json")
RETRIEVAL_TOP_K = 3
RETRIEVAL_MIN_SCORE = 0.15

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "what", "who", "when", "where", "why", "how", "i", "you", "we",
    "it", "to", "of", "in", "on", "for", "and", "or", "that", "this",
    "my", "our", "your", "about", "with",
}

SUMMARY_SYSTEM_PROMPT = (
    "You are compacting a conversation for reuse. Read the conversation "
    "below and produce a compact bullet list of durable facts, decisions, "
    "preferences, and open questions. Omit small talk and anything "
    "resolved or no longer relevant. Do not add commentary. Output only "
    "the bullet list."
)

FACT_EXTRACTION_PROMPT = (
    "Read the conversation below. Extract a JSON array of short, atomic, "
    "self-contained facts worth remembering long-term (decisions, names, "
    "numbers, project details, stated preferences). Each element must be "
    "a single string. Skip anything trivial or already obvious. Output "
    "ONLY the JSON array, nothing else."
)

# ---------- TTS Utilities (macOS Native) ----------

class TextToSpeech:
    def __init__(self, voice: str = "Samantha (Enhanced)"):
        self.enabled = True 
        self.voice = voice

    def sanitize_for_speech(self, text: str) -> str:
        """Strips markdown and prepends a 250ms silent buffer for CoreAudio warmup."""
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", " Code block omitted. ", text)
        # Remove inline code ticks
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove markdown emphasis (* and _)
        text = re.sub(r"[*_]{1,3}", "", text)
        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "link omitted", text)
        
        clean = text.strip()
        if not clean:
            return ""

        # [[slnc 250]] injects 250ms of silence at speech onset
        return f"[[slnc 500]] {clean}"


    def speak(self, text: str):
        if not self.enabled or not text:
            return
        
        clean_text = self.sanitize_for_speech(text)
        if not clean_text:
            return

        cmd = ["say"]
        if self.voice:
            cmd.extend(["-v", self.voice])
            
        # [[slnc 300]] provides 300ms of speech engine pad
        cmd.append(f"[[slnc 500]] {clean_text}")

        try:
            # Spawn non-blocking subprocess
            subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("[TTS Warning]: 'say' command not found.")
            self.enabled = False

# ---------- Ollama API Interaction ----------

def call_ollama(messages, model=MODEL):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------- Compaction & Fact Extraction ----------

def summarize(messages_to_compact):
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages_to_compact
    )
    summary_messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    return call_ollama(summary_messages)


def load_facts():
    if FACTS_PATH.exists():
        try:
            return json.loads(FACTS_PATH.read_text())
        except json.JSONDecodeError:
            return []
    return []


def save_facts(facts):
    FACTS_PATH.write_text(json.dumps(facts, indent=2))


def extract_facts(messages_to_compact):
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages_to_compact
    )
    extraction_messages = [
        {"role": "system", "content": FACT_EXTRACTION_PROMPT},
        {"role": "user", "content": transcript},
    ]
    raw = call_ollama(extraction_messages)
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        facts = json.loads(cleaned)
        return [f for f in facts if isinstance(f, str) and f.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def retrieve_relevant_facts(query, facts, top_k=RETRIEVAL_TOP_K):
    query_words = tokenize(query)
    if not query_words:
        return []

    scored = []
    for fact in facts:
        fact_words = tokenize(fact)
        if not fact_words:
            continue
        overlap = query_words & fact_words
        score = len(overlap) / len(query_words)
        if score >= RETRIEVAL_MIN_SCORE:
            scored.append((score, fact))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [fact for _, fact in scored[:top_k]]


def maybe_compact(history):
    user_turns = sum(1 for m in history if m["role"] == "user")
    if user_turns < SUMMARIZE_EVERY or len(history) <= KEEP_LAST:
        return history

    to_compact = history[:-KEEP_LAST]
    to_keep = history[-KEEP_LAST:]

    fact_summary = summarize(to_compact)
    new_facts = extract_facts(to_compact)

    if new_facts:
        facts = load_facts()
        facts.extend(new_facts)
        save_facts(facts)

    compacted = [
        {"role": "system", "content": f"Conversation summary so far:\n{fact_summary}"}
    ]
    compacted.extend(to_keep)
    return compacted


# ---------- Main CLI Loop ----------

def chat_loop():
    tts = TextToSpeech(voice="Samantha (Enhanced)")
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    history = []

    print(f"Chatting with {MODEL} - type 'quit' to exit.")
    print("Commands: /voice, /voice-off, /set-voice <voice_name>\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        # Handle system/flag commands
        cmd = user_input.lower()
        if cmd in {"quit", "exit"}:
            break
        elif cmd in {"/voice", "/say"}:
            tts.enabled = True
            print(f"[*] Voice mode enabled (Voice: {tts.voice})\n")
            continue
        elif cmd in {"/voice-off", "/say-off"}:
            tts.enabled = False
            print("[*] Voice mode disabled\n")
            continue
        elif cmd.startswith("/set-voice "):
            new_voice = user_input.split(maxsplit=1)[1].strip()
            tts.voice = new_voice
            print(f"[*] Voice changed to: {tts.voice}\n")
            continue

        history.append({"role": "user", "content": user_input})
        history = maybe_compact(history)

        facts = load_facts()
        hits = retrieve_relevant_facts(user_input, facts)

        messages = [system_prompt]
        if hits:
            recall_note = "Relevant facts from earlier in this project:\n" + "\n".join(
                f"- {f}" for f in hits
            )
            messages.append({"role": "system", "content": recall_note})
        messages.extend(history)

        reply = call_ollama(messages)
        history.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}\n")
        tts.speak(reply)


if __name__ == "__main__":
    chat_loop()
