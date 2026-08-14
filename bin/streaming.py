"""
Ollama Chat Loop with Real-Time Sentence Streaming TTS
- Uses httpx.AsyncClient to stream tokens from Ollama.
- Buffers tokens into sentences and enqueues them for sequential TTS.
- Includes speech-sanitization and macOS CoreAudio startup padding [[slnc 200]].

Requires: pip install httpx
Assumes Ollama is running locally: `ollama serve`
"""

import asyncio
import json
import queue
import re
import subprocess
import threading
import sys
import httpx
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "hermes3:latest"  # Fits well within 16GB Unified RAM

SUMMARIZE_EVERY = 6
KEEP_LAST = 4
FACTS_PATH = Path("facts.json")


# ---------- Streaming Text-to-Speech Engine ----------

class StreamingTTS:
    def __init__(self, voice: str = "Samantha (Enhanced)"):
        self.enabled = True
        self.voice = voice
        self.speech_queue = queue.Queue()
        self.is_running = True
        # Start a dedicated worker thread for sequential playback
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def sanitize_for_speech(self, text: str) -> str:
        """Strips markdown formatting and adds silence padding for CoreAudio initialization."""
        text = re.sub(r"```[\s\S]*?```", " Code block omitted. ", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"[*_]{1,3}", "", text)
        text = re.sub(r"http[s]?://\S+", "link omitted", text)
        
        clean = text.strip()
        if not clean:
            return ""
        # [[slnc 200]] prevents CoreAudio clipping on speech onset
        return f"[[slnc 200]] {clean}"

    def enqueue_sentence(self, text: str):
        if self.enabled and text.strip():
            self.speech_queue.put(text)

    def _speech_worker(self):
        """Processes enqueued sentences sequentially to prevent audio overlaps."""
        while self.is_running:
            try:
                # Block until a sentence is available
                sentence = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            clean_text = self.sanitize_for_speech(sentence)
            if clean_text:
                cmd = ["say"]
                if self.voice:
                    cmd.extend(["-v", self.voice])
                cmd.append(clean_text)

                # Use subprocess.run to wait for the sentence to finish playing before starting the next
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            self.speech_queue.task_done()

    def stop_current_speech(self):
        """Clears pending speech queue and stops audio."""
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()


# ---------- Async Streaming Ollama Client ----------

async def stream_ollama_chat(
    messages: list[dict[str, str]], 
    tts: StreamingTTS, 
    model: str = MODEL
) -> str:
    """Streams response from Ollama, printing tokens live and enqueuing complete sentences to TTS."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }

    full_response = []
    sentence_buffer = ""
    # Regex split on sentence terminators followed by space/newline
    sentence_end_pattern = re.compile(r'([.!?]+(?:\s+|\n+|$))')

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()

            async for chunk in response.aiter_lines():
                if not chunk:
                    continue
                
                try:
                    data = json.loads(chunk)
                    token = data.get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    continue

                if token:
                    # Print token directly to terminal
                    sys.stdout.write(token)
                    sys.stdout.flush()

                    full_response.append(token)
                    sentence_buffer += token

                    # Check if buffer contains a complete sentence
                    parts = sentence_end_pattern.split(sentence_buffer)
                    if len(parts) > 1:
                        # Reconstruct completed sentences
                        completed_sentence = "".join(parts[:-1])
                        tts.enqueue_sentence(completed_sentence)
                        sentence_buffer = parts[-1]  # Keep trailing fragment

    # Flush any remaining text left in the buffer
    if sentence_buffer.strip():
        tts.enqueue_sentence(sentence_buffer)

    print()  # Final newline for terminal
    return "".join(full_response)


# ---------- Main Async Chat Loop ----------

async def main():
    tts = StreamingTTS(voice="Samantha (Enhanced)")
    system_prompt = {"role": "system", "content": "You are a helpful, concise AI assistant."}
    history = []

    print(f"Chatting with {MODEL} (Streaming Enabled)")
    print("Commands: /voice, /voice-off, /set-voice <voice_name>, quit\n")

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
        elif cmd in {"/voice", "/say"}:
            tts.enabled = True
            print(f"[*] Voice mode enabled (Voice: {tts.voice})\n")
            continue
        elif cmd in {"/voice-off", "/say-off"}:
            tts.enabled = False
            tts.stop_current_speech()
            print("[*] Voice mode disabled\n")
            continue
        elif cmd.startswith("/set-voice "):
            new_voice = user_input.split(maxsplit=1)[1].strip()
            tts.voice = new_voice
            print(f"[*] Voice changed to: {tts.voice}\n")
            continue

        history.append({"role": "user", "content": user_input})

        messages = [system_prompt] + history
        
        print("\nAssistant: ", end="", flush=True)
        reply = await stream_ollama_chat(messages, tts)
        history.append({"role": "assistant", "content": reply})
        print()

if __name__ == "__main__":
    asyncio.run(main())
