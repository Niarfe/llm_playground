import pyttsx3
import time

engine = pyttsx3.init()

# Keep audio channel hot without blocking execution
engine.startLoop(False) 

# Whenever you want to speak, queue it and manually pump the driver
def live_speak(text):
    engine.say(text)
    engine.iterate() # Forces immediate driver processing

# Test it
live_speak("This will play immediately without clipping.")
time.sleep(5)
live_speak("The hardware is still awake, so this will not drop syllables either.")

# Clean up when your main program shuts down
engine.endLoop() 

