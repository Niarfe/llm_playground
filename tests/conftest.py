"""
Load the numbered example modules by path.

The examples are named `03_compaction.py` and so on, which is good for
reading order but not a valid Python identifier, so a normal import will
not work. importlib loads them by file path instead.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(filename: str):
    path = EXAMPLES / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def compaction():
    return load("03_compaction.py")


@pytest.fixture(scope="session")
def memory():
    return load("04_fact_memory.py")


@pytest.fixture(scope="session")
def tts():
    return load("05_tts_say.py")


@pytest.fixture(scope="session")
def streaming():
    return load("06_streaming_tts.py")


@pytest.fixture(scope="session")
def tools():
    return load("07_tool_calling.py")
