"""
Load the numbered example modules by path.

The examples are named `04_compaction.py` and so on, which is good for
reading order but not a valid Python identifier, so a normal import will
not work. importlib loads them by file path instead.

01 is deliberately absent: it runs its calls at module level (it is a
script, not a module), so importing it would talk to Ollama. Everything
tested here guards its entry point with `if __name__ == "__main__"`.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def load(relative_path: str):
    path = EXAMPLES / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def compaction():
    return load("04_compaction.py")


@pytest.fixture(scope="session")
def memory():
    return load("05_fact_memory.py")


@pytest.fixture(scope="session")
def tools():
    return load("06_tool_calling.py")


@pytest.fixture(scope="session")
def retry():
    return load("07_retry_loop.py")


@pytest.fixture(scope="session")
def agent():
    return load("08_agent_loop.py")


@pytest.fixture(scope="session")
def tts():
    return load("extras/tts_say.py")


@pytest.fixture(scope="session")
def streaming():
    return load("extras/streaming_tts.py")
