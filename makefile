PYTHON := env/bin/python

default:
	@cat makefile

env:
	python3 -m venv env; . env/bin/activate; pip install --upgrade pip

update: env
	. env/bin/activate; pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -q

# Examples. Each needs `ollama serve` running in another terminal.
run-01:
	$(PYTHON) examples/01_minimal_chat.py

run-02:
	$(PYTHON) examples/02_pick_model_and_prompt.py

run-03:
	$(PYTHON) examples/03_compaction.py

run-04:
	$(PYTHON) examples/04_fact_memory.py

run-05:
	$(PYTHON) examples/05_tts_say.py

run-06:
	$(PYTHON) examples/06_streaming_tts.py

run-07:
	$(PYTHON) examples/07_tool_calling.py

run-08:
	$(PYTHON) examples/08_structured_reasoning.py

.PHONY: default env update test run-01 run-02 run-03 run-04 run-05 run-06 run-07 run-08
