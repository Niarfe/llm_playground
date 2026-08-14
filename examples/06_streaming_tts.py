"""
06 — Speak while the model is still generating.

THE PROBLEM
    Example 05 speaks a finished reply. But a local model on a laptop
    produces maybe 20 tokens a second, so a long answer means ten seconds
    of silence before a single word is spoken. It feels broken.

THE FIX, IN TWO PARTS
    1. BUFFER TOKENS INTO SENTENCES. Tokens arrive as fragments ("The",
       " qu", "ick"). You cannot speak a fragment. Accumulate until you
       see sentence-ending punctuation, then release that sentence.

    2. SPEAK ON A WORKER THREAD. Speech is far slower than generation, so
       the two must be decoupled. A queue.Queue between them means the
       generator never waits on audio, and a single consumer thread
       guarantees sentences play in order and never overlap.

           network -> token loop -> [sentence] -> Queue -> worker -> say

    That queue is doing real work: it is both the buffer and the lock.

THE TUNING KNOB
    MIN_BUFFER_LENGTH stops tiny fragments ("Sure!") from each spawning
    their own `say` process, which produces audible stutter between
    clauses. Raising it means smoother speech but a longer initial delay.
    This is the whole tradeoff -- there is no setting that wins both.

RUN IT
    python examples/06_streaming_tts.py

    Commands: /voice  /voice-off  /set-voice <name>  quit
"""

import asyncio
import json
import queue
import re
import subprocess
import sys
import threading

import httpx

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "dolphin3"

# Below this many characters, hold the sentence back and let it merge with
# the next one rather than spawning a `say` process for a few words.
MIN_BUFFER_LENGTH = 40


class StreamingTTS:
    def __init__(self, voice: str = "Samantha"):
        self.enabled = True
        self.voice = voice
        self.speech_queue = queue.Queue()
        self.is_running = True

        # daemon=True so a stuck worker cannot keep the process alive on exit
        self.worker = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker.start()

    def sanitize(self, text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", " Code block omitted. ", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"[*_]{1,3}", "", text)
        text = re.sub(r"http[s]?://\S+", "link omitted", text)
        clean = text.strip()
        return f"[[slnc 200]] {clean}" if clean else ""

    def enqueue(self, text: str) -> None:
        if self.enabled and text.strip():
            self.speech_queue.put(text.strip())

    def _speech_worker(self) -> None:
        """
        Drain the queue and speak. Runs forever on its own thread.

        Note it collects every chunk currently waiting and merges them into
        ONE `say` invocation. Speaking them separately leaves an audible gap
        at each process boundary; merging makes it sound continuous.
        """
        while self.is_running:
            try:
                first = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue  # timeout, not an error -- lets us re-check is_running

            chunks = [first]
            self.speech_queue.task_done()

            while not self.speech_queue.empty():
                try:
                    chunks.append(self.speech_queue.get_nowait())
                    self.speech_queue.task_done()
                except queue.Empty:
                    break

            clean = self.sanitize(" ".join(chunks))
            if not clean:
                continue

            cmd = ["say"]
            if self.voice:
                cmd.extend(["-v", self.voice])
            cmd.append(clean)

            # Blocking on purpose: this thread must not start the next
            # utterance until this one has finished playing.
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_current_speech(self) -> None:
        """Drop anything still queued. Does not interrupt what is playing."""
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()


SENTENCE_END = re.compile(r"([.!?]+(?:\s+|\n+|$))")


def take_completed_sentences(buffer: str, min_length: int = MIN_BUFFER_LENGTH):
    """
    Split off whatever complete sentences the buffer holds.

    Returns (speakable, remainder). If nothing is ready -- no punctuation
    yet, or too short to be worth its own `say` process -- returns
    ("", buffer) and the caller keeps accumulating.
    """
    parts = SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return "", buffer

    candidate = "".join(parts[:-1])
    if len(candidate.strip()) < min_length:
        return "", buffer

    return candidate, parts[-1]


async def stream_chat(messages, tts, model=MODEL) -> str:
    """Stream tokens from Ollama; print each, speak each completed sentence."""
    payload = {"model": model, "messages": messages, "stream": True}

    full_response = []
    buffer = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()

            # Ollama streams newline-delimited JSON, one object per token.
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    token = json.loads(line).get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    continue
                if not token:
                    continue

                sys.stdout.write(token)
                sys.stdout.flush()

                full_response.append(token)
                buffer += token

                speakable, buffer = take_completed_sentences(buffer)
                if speakable:
                    tts.enqueue(speakable)

    if buffer.strip():
        tts.enqueue(buffer)  # flush whatever never hit punctuation

    print()
    return "".join(full_response)


async def main():
    tts = StreamingTTS(voice="Samantha")
    system_prompt = {
        "role": "system",
        "content": "You are a helpful, concise assistant. Do not use emoji, "
        "since responses are read aloud. Keep responses short.",
    }
    history = []

    print(f"Chatting with {MODEL} (streaming + TTS)")
    print("Commands: /voice, /voice-off, /set-voice <name>, quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in {"quit", "exit"}:
            break
        if cmd in {"/voice", "/say"}:
            tts.enabled = True
            print(f"[*] Voice on ({tts.voice})\n")
            continue
        if cmd in {"/voice-off", "/say-off"}:
            tts.enabled = False
            tts.stop_current_speech()
            print("[*] Voice off\n")
            continue
        if cmd.startswith("/set-voice "):
            tts.voice = user_input.split(maxsplit=1)[1].strip()
            print(f"[*] Voice set to {tts.voice}\n")
            continue

        history.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        reply = await stream_chat([system_prompt] + history, tts)
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    asyncio.run(main())
