import threading
import queue
import pyttsx3
import time

# 1. Thread-safe queue for incoming sentences from Ollama
speech_queue = queue.Queue()

def ollama_llm_worker():
    """
    This background thread handles your Ollama client generation.
    It streams tokens, packages them into sentences, and pushes them to the queue.
    """
    print("[Ollama Thread] Simulating Ollama response generation...")
    
    responses = [
        "This first sentence will now play completely without cutting off.",
        "Because the speech engine is anchored to the macOS main thread, the Cocoa loop stays alive.",
        "Your audio hardware will remain hot, entirely eliminating the first syllable clipping issue."
    ]
    
    for sentence in responses:
        speech_queue.put(sentence)
        time.sleep(4)  # Simulate the time it takes Ollama to think/generate
        
    # Send a sentinel value to tell the main thread it's time to stop listening
    speech_queue.put(None)

if __name__ == "__main__":
    # 2. Start your Ollama agent in the background
    ollama_thread = threading.Thread(target=ollama_llm_worker, daemon=True)
    ollama_thread.start()

    # 3. Initialize and run the TTS engine explicitly on the MAIN THREAD
    print("[Main Thread] Initializing persistent TTS engine...")
    engine = pyttsx3.init()
    engine.setProperty('rate', 185) # Tweak speed if desired
    
    while True:
        # Pull text from the queue (blocks until a sentence arrives)
        text = speech_queue.get()
        
        # If we receive the shutdown signal, exit the loop cleanly
        if text is None:
            break
            
        if text.strip():
            print(f"[Speaking]: {text}")
            engine.say(text)
            engine.runAndWait() # This blocks safely now because it's on the main thread
            
    print("Ollama stream finished and all text successfully spoken.")

