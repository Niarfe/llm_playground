"""
Tests for 06_tool_calling.py.

The interesting part is not that the tool runs -- it is that the tool
refuses. A tool is a security boundary, so the boundary is what gets tested.
"""


def test_runs_the_fixture_script(tools):
    output = tools.execute_local_script("hello.py")
    assert "Hello from a script the model asked to run." in output


def test_missing_file_reports_cleanly(tools):
    assert "not found" in tools.execute_local_script("no_such_script.py").lower()


def test_refuses_parent_directory_escape(tools):
    assert "refused" in tools.execute_local_script("../../../etc/hosts").lower()


def test_refuses_absolute_path_outside_scripts(tools):
    assert "refused" in tools.execute_local_script("/etc/passwd").lower()


def test_refuses_a_sibling_example(tools):
    """
    The regression this directory move was made for.

    When the boundary was examples/ rather than examples/scripts/, this
    resolved inside the allowed directory and would have launched an
    interactive chat loop under the model's direction.
    """
    result = tools.execute_local_script("../04_compaction.py")
    assert "refused" in result.lower()


def test_allows_paths_inside_scripts(tools):
    """The boundary must not be so tight that the tool is useless."""
    assert "refused" not in tools.execute_local_script("./hello.py").lower()
