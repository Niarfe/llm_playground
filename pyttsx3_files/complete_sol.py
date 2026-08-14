import threading
import queue
import pyttsx3
import time

# 1. Thread-safe queue for incoming sentences from Ollama
speech_queue = queue.Queue()

def ollama_llm_worker():
    """
    Background worker simulating your Ollama stream.
    It passes strings to the queue as they finish generating.
    """
    print("[Ollama Thread] Simulating Ollama response generation...")
    
    responses = [
        "This first sentence plays completely without cutting off.",
        "And now the second sentence plays right after it successfully.",
        "The third sentence will also play without stalling the loop."
    ]
    
    for sentence in responses:
        speech_queue.put(sentence)
        time.sleep(4)  # Simulate model processing delay
        
    # Send shutdown signal to cleanly exit the main thread loop
    speech_queue.put(None)

if __name__ == "__main__":
    # 2. Fire up the Ollama background process
    ollama_thread = threading.Thread(target=ollama_llm_worker, daemon=True)
    ollama_thread.start()

    print("[Main Thread] Listening for sentences from the queue...")
    
    while True:
        # Pull text from the queue (blocks cleanly until text arrives)
        text = speech_queue.get()
        
        if text is None:
            break  # Quit signal received
            
        if text.strip():
            print(f"[Speaking]: {text}")
            
            # --- THE MAC WORKAROUND CRUX ---
            # 1. Spawn a fresh engine instance to fix the subsequent silence bug
            engine = pyttsx3.init()
            engine.setProperty('rate', 185)
            
            # 2. Prepend a 600ms silence macro so the audio card wakes up safely
            padded_text = f"[[slnc 600]] {text}"
            
            # 3. Say and run cleanly on the Main Thread
            engine.say(padded_text)
            engine.runAndWait()
            # -------------------------------
            
        speech_queue.task_done()
                
    print("Stream finished. Audio context torn down cleanly.")

