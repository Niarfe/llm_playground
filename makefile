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

# 09: fix a failing test. The test suite is the verifier.
run-09:
	$(PYTHON) examples/09_debugging_agent.py --scenario 1

run-09-case2:
	$(PYTHON) examples/09_debugging_agent.py --scenario 2

run-09-case3:
	$(PYTHON) examples/09_debugging_agent.py --scenario 3

# All three bugs. Each lives in a different file, so the path differs.
run-09-all:
	-$(PYTHON) examples/09_debugging_agent.py --scenario 1
	-$(PYTHON) examples/09_debugging_agent.py --scenario 2
	-$(PYTHON) examples/09_debugging_agent.py --scenario 3

run-09-all-qwen:
	-$(PYTHON) examples/09_debugging_agent.py --scenario 1 --model qwen2.5:7b-instruct
	-$(PYTHON) examples/09_debugging_agent.py --scenario 2 --model qwen2.5:7b-instruct
	-$(PYTHON) examples/09_debugging_agent.py --scenario 3 --model qwen2.5:7b-instruct

# 10: lint, then tests, then a reviewer. Each tier catches a different bug.
run-10:
	$(PYTHON) examples/10_layered_review.py

run-10-coder:
	$(PYTHON) examples/10_layered_review.py --model qwen2.5-coder:7b

# ---- deepenings: a better part in an existing slot ----
# Written but NOT yet verified end to end. Run them and report.

run-05a:
	$(PYTHON) examples/05a_rag.py

run-07a:
	$(PYTHON) examples/07a_structured_output.py

# ---- extras: optional, off the main line ----

run-tts:
	$(PYTHON) examples/extras/tts_say.py

run-tts-streaming:
	$(PYTHON) examples/extras/streaming_tts.py

run-reasoning:
	$(PYTHON) examples/extras/structured_reasoning.py

.PHONY: default env update models models-optional test \
	run-01 run-02 run-03 run-04 run-05 run-06 run-07 run-08 run-08-gated run-09 run-09-case2 run-09-case3 run-09-all run-09-all-qwen run-10 run-10-coder run-05a run-07a \
	run-tts run-tts-streaming run-reasoning
