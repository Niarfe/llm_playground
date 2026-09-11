"""
05a -- Retrieval that understands meaning. (This is what "RAG" means.)

STATUS: PARTIALLY VERIFIED
    The retrieval comparison below was measured directly (bge-m3, cosine,
    2026-09-11) and those three results are real. What has NOT been run is
    this file end to end. Run it; the numbers should regenerate, because the
    example IS the measurement.

WHERE THIS ATTACHES
    A deepening of 05, not a step past it. 05 retrieves facts by counting
    shared words, and its own docstring promises this file:

        "reach for embeddings when you can name a query this gets wrong"

    tests/test_fact_memory.py then names two, and they have been sitting
    there as PASSING tests -- pinned failures, waiting for a fix:

      test_known_limitation_synonyms_miss
      test_known_limitation_incidental_words_cause_false_hits

MAIN POINT
    An embedding turns text into a list of numbers positioned so that things
    which MEAN similar things land near each other. Retrieval then becomes
    geometry: embed the question, find the nearest facts.

    That is the whole idea. "RAG" is retrieval-augmented generation, and the
    augmentation half you already built in 05 -- inject what you found into
    the prompt. Only the retrieval half changes here.

WHY THERE IS NO VECTOR DATABASE HERE
    Cosine similarity is the `cosine` function below: eight lines, no
    dependencies. That is genuinely all it is.

    FAISS, Chroma, pgvector and the rest solve a different problem --
    approximate nearest-neighbour search when exact comparison against every
    vector stops being free. That happens somewhere north of a hundred
    thousand vectors. With six facts, exact search costs nothing, and
    importing an index would hide the one mechanism this file exists to show.

    Reach for a vector store when you can say how many vectors you have and
    why the loop is too slow. Not before.

WHAT EMBEDDINGS ARE WORSE AT -- MEASURED
    The honest half, and the reason production systems run BOTH retrievers:

      "Describe the feline"      keyword: nothing      embeddings: correct
      "How old is the feline?"   keyword: right, but only by matching "old"
      "status of invoice 4471?"  keyword: correct      embeddings: WRONG

    That last one ranked invoice 4472 above 4471, by 0.758 to 0.756.
    Semantic similarity has no notion of exact identity -- an order number,
    a date, a version string, an error code. And it fails by a hair, which
    is worse than failing loudly.

    Keyword search is exact and literal. Embeddings are approximate and
    semantic. Neither dominates. Hybrid retrieval is not a buzzword, it is
    an admission that these two failure modes do not overlap.

PREREQUISITE
    ollama pull bge-m3

RUN IT
    python examples/05a_rag.py
"""

import math
import re

import ollama

EMBED_MODEL = "bge-m3"
TOP_K = 3

FACTS = [
    "The user's cat is named Blue.",
    "Blue the cat is 4 years old.",
    "The project deadline is March 14th.",
    "The user prefers dark mode.",
    "Invoice 4471 was paid on the 2nd.",
    "Invoice 4472 is still outstanding.",
]

QUERIES = [
    ("Describe the feline", "synonym -- keyword cannot know feline means cat"),
    ("How old is the feline?", "keyword gets this right, but only via 'old'"),
    ("What is the status of invoice 4471?", "exact identifier -- embeddings lose"),
]


##############################################################################
# KEYWORD RETRIEVAL                                         [extended from 05]
##############################################################################
# 05's retriever, with retrieve_relevant_facts renamed to keyword_search so
# the two methods read side by side. Logic is untouched; the rename is why
# this is tagged 'extended' rather than 'unchanged' -- tests/test_provenance.py
# checks that claim byte for byte, and caught the mislabel.

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "what", "who", "when", "where", "why", "how", "i", "you", "we",
    "it", "to", "of", "in", "on", "for", "and", "or", "that", "this",
    "my", "our", "your", "about", "with",
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def keyword_search(query, facts, top_k=TOP_K):
    query_words = tokenize(query)
    if not query_words:
        return []
    scored = []
    for fact in facts:
        fact_words = tokenize(fact)
        if not fact_words:
            continue
        score = len(query_words & fact_words) / len(query_words)
        if score >= 0.15:
            scored.append((score, fact))
    scored.sort(reverse=True)
    return scored[:top_k]


##############################################################################
# EMBEDDING RETRIEVAL                                           [new in 05a]
##############################################################################
# The entire mechanism. There is no more to it than this.

def embed(texts):
    """Text in, lists of floats out. One vector per input string."""
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def cosine(a, b):
    """
    How aligned two vectors are, ignoring their length. 1.0 is identical
    direction, 0.0 is unrelated.

    Length is ignored on purpose: a long fact and a short one about the same
    subject should score alike, and only the direction carries the meaning.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def semantic_search(query, facts, fact_vectors, top_k=TOP_K):
    """Embed the query, compare against every fact, return the nearest."""
    query_vector = embed([query])[0]
    scored = [(cosine(query_vector, v), f) for v, f in zip(fact_vectors, facts)]
    scored.sort(reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    print(f"[embedding model] {EMBED_MODEL}")

    fact_vectors = embed(FACTS)
    print(f"[vectors] {len(fact_vectors)} facts, {len(fact_vectors[0])} dimensions each")
    print("[index] a Python list. Exact search over 6 vectors is instant.\n")

    for query, note in QUERIES:
        print("=" * 74)
        print(f"Q: {query}")
        print(f"   ({note})")
        print("=" * 74)

        kw = keyword_search(query, FACTS)
        print("  keyword:")
        if kw:
            for score, fact in kw:
                print(f"    {score:.3f}  {fact}")
        else:
            print("    (nothing matched)")

        print("  embeddings:")
        for score, fact in semantic_search(query, FACTS, fact_vectors)[:2]:
            print(f"    {score:.3f}  {fact}")
        print()

    print(
        "Both retrievers, same facts, same questions. Neither wins outright.\n"
        "The synonym query is hopeless for keyword matching; the invoice query\n"
        "is hopeless for embeddings, and they miss it by a whisker rather than\n"
        "obviously. Run both and merge the results -- that is hybrid retrieval,\n"
        "and this is the argument for it.\n"
    )
