PYTHON := env/bin/python

default:
	@cat makefile

env:
	python3 -m venv env; . env/bin/activate; pip install --upgrade pip

update: env
	. env/bin/activate; pip install -r requirements.txt

models:
	ollama pull llama3.1
	ollama pull qwen2.5:1.5b-instruct

models-optional:
	ollama pull qwen2.5:7b-instruct
	ollama pull deepseek-r1:8b

test:
	$(PYTHON) -m pytest tests/ -q

# ---- the main line: read and run these in order ----
# Each needs `ollama serve` running in another terminal.

run-01:
	$(PYTHON) examples/01_minimal_chat.py

run-02:
	$(PYTHON) examples/02_pick_model_and_prompt.py

run-03:
	$(PYTHON) examples/03_temperature.py

run-04:
	$(PYTHON) examples/04_compaction.py

run-05:
	$(PYTHON) examples/05_fact_memory.py

run-06:
	$(PYTHON) examples/06_tool_calling.py

run-07:
	$(PYTHON) examples/07_retry_loop.py

run-08:
	$(PYTHON) examples/08_agent_loop.py

# Same example, with tools gated so the model cannot speculate on turn 1.
run-08-gated:
	$(PYTHON) examples/08_agent_loop.py --gated

# Experimental: a task whose PATH branches. Three cases, three routes.
run-09:
	$(PYTHON) examples/09_branching_agent.py

run-09-case2:
	$(PYTHON) examples/09_branching_agent.py --case 2

run-09-case3:
	$(PYTHON) examples/09_branching_agent.py --case 3

# ---- extras: optional, off the main line ----

run-tts:
	$(PYTHON) examples/extras/tts_say.py

run-tts-streaming:
	$(PYTHON) examples/extras/streaming_tts.py

run-reasoning:
	$(PYTHON) examples/extras/structured_reasoning.py

.PHONY: default env update models models-optional test \
	run-01 run-02 run-03 run-04 run-05 run-06 run-07 run-08 run-08-gated run-09 run-09-case2 run-09-case3 \
	run-tts run-tts-streaming run-reasoning
