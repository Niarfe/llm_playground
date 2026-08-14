import threading
import queue
import pyttsx3
import time

# 1. Thread-safe queue for incoming sentences from Ollama
speech_queue = queue.Queue()

def ollama_llm_worker():
    """
    Background worker that fetches output from Ollama.
    """
    print("[Ollama Thread] Simulating Ollama response generation...")
    
    responses = [
        "This first sentence plays completely without cutting off.",
        "And now the second sentence plays right after it successfully.",
        "The third sentence will also play without clipping any audio context."
    ]
    
    for sentence in responses:
        speech_queue.put(sentence)
        time.sleep(4)  # Simulate model thinking time
        
    # Send a shutdown signal to exit the main engine loop cleanly
    speech_queue.put(None)

if __name__ == "__main__":
    # 2. Fire up the Ollama background script
    ollama_thread = threading.Thread(target=ollama_llm_worker, daemon=True)
    ollama_thread.start()

    # 3. Handle TTS via an external manual loop tracker on the MAIN THREAD
    print("[Main Thread] Initializing persistent manual event loop...")
    engine = pyttsx3.init()
    engine.setProperty('rate', 185)
    
    # Passing False flags pyttsx3 that we will manually pump events
    engine.startLoop(False)
    
    try:
        while True:
            # Continuously pump the Cocoa driver to flush out any sound buffers
            engine.iterate()
            
            # Check the queue without blocking so the loop keeps iterating
            try:
                text = speech_queue.get_nowait()
                
                if text is None:
                    break  # Quit signature received
                    
                if text.strip():
                    print(f"[Speaking]: {text}")
                    engine.say(text)
                    
                    # Instead of blocking the whole thread with runAndWait(),
                    # keep pumping events manually while the speech engine is actively busy
                    while engine.isBusy():
                        engine.iterate()
                        time.sleep(0.02) # Low sleep interval protects CPU cycles
                        
                speech_queue.task_done()
                
            except queue.Empty:
                # If nothing is in the queue, sleep briefly and loop back to keep context alive
                time.sleep(0.1)
                
    finally:
        # Tear down loop gracefully when thread finishes
        engine.endLoop()
        print("Stream finished. Audio context torn down cleanly.")

