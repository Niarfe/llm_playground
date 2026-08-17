"""
The duplication guard.

This repo copies code between examples on purpose, so each file reads top to
bottom without chasing an import. The cost of that choice is drift: a block
labelled `[unchanged from 02]` quietly stops being unchanged.

These tests make the label enforceable. A block claiming to be unchanged must
be byte-identical to the block it names. If it drifts, the failure says which
two files disagree.

Banner format:

    ##############################################################
    # SECTION TITLE                              [unchanged from 02]
    ##############################################################
    # optional context lines

Tags: [new in NN], [unchanged from NN], [extended from NN].
Only `unchanged from` is enforced -- `extended from` is an explicit signal
that the block was modified on purpose.
"""

import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

RULE = re.compile(r"^#{20,}$")
TITLE = re.compile(r"^#\s+(?P<title>.*?)\s*(?:\[(?P<tag>[^\]]+)\])?\s*$")


def example_files():
    return sorted(EXAMPLES.glob("*.py")) + sorted(EXAMPLES.glob("extras/*.py"))


def parse_blocks(path: Path):
    """Return {title: (tag, body)} for every banner block in a file."""
    lines = path.read_text().splitlines()
    blocks = {}
    current = None
    body: list[str] = []

    index = 0
    while index < len(lines):
        is_banner = (
            RULE.match(lines[index])
            and index + 2 < len(lines)
            and RULE.match(lines[index + 2])
            and TITLE.match(lines[index + 1])
        )

        if is_banner:
            if current:
                blocks[current[0]] = (current[1], "\n".join(body).strip())
            match = TITLE.match(lines[index + 1])
            current = (match.group("title").strip(), match.group("tag"))
            body = []
            index += 3
            continue

        if current:
            body.append(lines[index].rstrip())
        index += 1

    if current:
        blocks[current[0]] = (current[1], "\n".join(body).strip())

    return blocks


ALL_BLOCKS = {path.name: parse_blocks(path) for path in example_files()}

UNCHANGED_CLAIMS = [
    (filename, title, tag)
    for filename, blocks in ALL_BLOCKS.items()
    for title, (tag, _) in blocks.items()
    if tag and tag.startswith("unchanged from")
]


def find_source(number: str):
    """Locate the example file whose name starts with the given number."""
    for filename in ALL_BLOCKS:
        if filename.startswith(f"{number}_"):
            return filename
    return None


def test_the_parser_found_something():
    """Guards against the banner format changing and silently disabling this."""
    total = sum(len(blocks) for blocks in ALL_BLOCKS.values())
    assert total >= 10, f"only parsed {total} banner blocks -- format drift?"


def test_at_least_one_unchanged_claim_exists():
    """If this fails the guard is passing vacuously."""
    assert UNCHANGED_CLAIMS, "no [unchanged from NN] blocks found to verify"


@pytest.mark.parametrize(
    "filename,title,tag",
    UNCHANGED_CLAIMS,
    ids=[f"{f}:{t}" for f, t, _ in UNCHANGED_CLAIMS],
)
def test_unchanged_blocks_really_are_unchanged(filename, title, tag):
    number = tag.split()[-1]
    source_name = find_source(number)
    assert source_name, f"{filename} claims '{tag}' but no example {number} exists"

    source_blocks = ALL_BLOCKS[source_name]
    assert title in source_blocks, (
        f"{filename} has a block '{title}' claiming '{tag}', but {source_name} "
        f"has no block with that title. Titles must match exactly. "
        f"{source_name} has: {sorted(source_blocks)}"
    )

    _, copied_body = ALL_BLOCKS[filename][title]
    _, source_body = source_blocks[title]

    assert copied_body == source_body, (
        f"'{title}' in {filename} has drifted from {source_name}.\n"
        f"Either re-sync it, or relabel it [extended from {number}] if the "
        f"change was deliberate."
    )


@pytest.mark.parametrize("filename", sorted(ALL_BLOCKS))
def test_tags_are_well_formed(filename):
    """Catch typos like [unchanged from 2] or [copied from 02]."""
    valid = re.compile(r"^(new in \d{2}|unchanged from \d{2}|extended from \d{2})$")
    for title, (tag, _) in ALL_BLOCKS[filename].items():
        if tag is None:
            continue
        assert valid.match(tag), (
            f"{filename}: block '{title}' has malformed tag '[{tag}]'. "
            f"Expected 'new in NN', 'unchanged from NN', or 'extended from NN'."
        )
