import threading
import queue
import time
# Import the native macOS AppKit speech synthesis engine
from AppKit import NSSpeechSynthesizer 

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
        "The second sentence now plays right after it with perfect sound.",
        "And the third sentence loops seamlessly without stalling the terminal."
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

    print("[Main Thread] Initializing Native Apple TTS Engine...")
    
    # 3. Instantiate the Mac speech synthesizer ONCE to keep the audio channel hot
    # This prevents the sleep-lag that causes initial syllable clipping.
    synth = NSSpeechSynthesizer.alloc().init()
    
    # Optional: Set a specific system voice if you don't want the default
    # synth.setVoice_("com.apple.speech.synthesis.voice.Alex")

    while True:
        # Pull text from the queue (blocks cleanly until text arrives)
        text = speech_queue.get()
        
        if text is None:
            break  # Quit signal received
            
        if text.strip():
            print(f"[Speaking]: {text}")
            
            # Pad the text with 200ms of native Apple silence to give the 
            # core audio layer an explicit buffer window before speaking
            padded_text = f"[[slnc 200]] {text}"
            
            # Trigger native speech asynchronously
            synth.startSpeakingString_(padded_text)
            
            # Explicitly poll the native hardware driver until it finishes speaking
            while synth.isSpeaking():
                time.sleep(0.05) # Yields CPU cycles without breaking the thread
                
        speech_queue.task_done()
                
    print("Stream finished. Audio context torn down cleanly.")

