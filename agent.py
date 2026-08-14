"""
Ollama Tool-Calling Script Executor
===================================
An agentic script that leverages Ollama's native tool-calling capability to 
allow an LLM to request and execute Python scripts on your local laptop.

Prerequisites:
-------------
1. Install Ollama Python SDK:
   $ pip install ollama

2. Download a tool-compatible model:
   $ ollama pull llama3.1

Usage:
------
1. Place the Python script you want to run (e.g., `hello.py`) in the workspace.
2. Edit the `messages` content in this file to instruct the LLM (e.g., "Run the script hello.py").
3. Execute this wrapper script:
   $ python agent.py

How it Works:
------------
The script passes the python function `execute_local_script` to Ollama as a tool.
The LLM returns a JSON instruction to call the tool, this script intercepts that 
instruction, runs the code locally using `subprocess`, and streams the console 
output back to the LLM.
"""

import os
from ollama import chat

# 1. Define the tool function (wrapper for local execution)
def execute_local_script(script_path: str):
    """Executes a local Python script and returns output."""
    if os.path.exists(script_path):
        import subprocess
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        return result.stdout or result.stderr
    return "File not found."

# 2. Setup agent and tool
messages = [{'role': 'user', 'content': 'Run the local script "./hello.py"'}]
tools = [execute_local_script]

# 3. Call Ollama with tools
response = chat(model='llama3.1', messages=messages, tools=tools)

# 4. Execute tool if requested by LLM
if response.message.tool_calls:
    for tool in response.message.tool_calls:
        if tool.function.name == 'execute_local_script':
            # Run on laptop and pass result back
            output = execute_local_script(**tool.function.arguments)
            messages.append(response.message)
            messages.append({'role': 'tool', 'content': output})
            
            # Get final response
            final = chat(model='llama3.1', messages=messages)
            print(final.message.content)

