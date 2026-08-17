"""
The judge for 09. Deterministic, no model, no opinions.

WHY THIS IS A SEPARATE FILE
    The agent in 09 produces a verdict and has no idea whether it is right.
    It does not import this file, and this file is the only thing that reads
    solutions.json. Run the agent all day: it can never tell you it passed.

    That separation is not ceremony. extras/structured_reasoning.py has a
    recorded run where a model checking its own arithmetic wrote "Consistent"
    under a result of 108 against its own stated bound of "less than 56". A
    system that grades its own work will eventually pass itself, confidently
    and in the correct output format.

    So: the agent writes an artifact, and something dumb and deterministic
    reads it. The dumbness is the feature.

WHAT COUNTS AS PASSING
    Two things, reported separately, because they fail independently:

      VERDICT   the right person was accused
      PATH      the right records were consulted to get there

    A run can pass the verdict and fail the path -- case 2 has done exactly
    that, reaching Verity via the alibi file rather than the guest book.
    Right answer, wrong route. Scoring only the verdict would have called
    that a clean pass and taught us nothing.

RUN IT
    env/bin/python examples/09_branching_agent.py --case 1
    env/bin/python examples/09_branching_agent.py --case 2
    env/bin/python examples/09_branching_agent.py --case 3
    env/bin/python examples/check_verdicts.py

    Or: make run-09-all
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLUTIONS = HERE / "solutions.json"
RUNS = HERE.parent / "runs"


def load_solutions() -> dict:
    data = json.loads(SOLUTIONS.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def judge(case_number: str, solution: dict) -> dict:
    artifact = RUNS / f"verdict-case-{case_number}.json"

    if not artifact.exists():
        return {"case": case_number, "status": "NOT RUN"}

    verdict = json.loads(artifact.read_text())
    accused = (verdict.get("accused") or "").strip()

    # Surname match, case-insensitive. The model may write "Colonel Ashby".
    named_right = bool(accused) and solution["culprit"].lower() in accused.lower()

    # Path check: were the expected records among those actually read? Extra
    # reads are tolerated; missing ones are not.
    read = set(verdict.get("files_read") or [])
    expected = set(solution["expected_path"])
    took_route = expected.issubset(read)

    return {
        "case": case_number,
        "status": "PASS" if named_right else "FAIL",
        "accused": accused or "(none)",
        "expected": solution["culprit"],
        "path_ok": took_route,
        "missing": sorted(expected - read),
        "extra": sorted(read - expected),
        "turns": verdict.get("turns"),
        "model": verdict.get("model", "?"),
    }


def main() -> int:
    solutions = load_solutions()
    results = [judge(number, solutions[number]) for number in sorted(solutions)]

    print()
    print("=" * 72)
    print("  09 VERDICTS -- checked against solutions.json, not by the agent")
    print("=" * 72)
    print(f"  {'CASE':<6}{'VERDICT':<9}{'ACCUSED':<14}{'EXPECTED':<10}"
          f"{'PATH':<7}{'TURNS':<7}MODEL")
    print("  " + "-" * 68)

    ran = [r for r in results if r["status"] != "NOT RUN"]

    for r in results:
        if r["status"] == "NOT RUN":
            print(f"  {r['case']:<6}{'NOT RUN':<9}"
                  f"{'-- run it first --':<34}")
            continue
        path = "ok" if r["path_ok"] else "WRONG"
        print(f"  {r['case']:<6}{r['status']:<9}{r['accused'][:13]:<14}"
              f"{r['expected']:<10}{path:<7}{str(r['turns']):<7}{r['model']}")

    if not ran:
        print("\n  Nothing to judge yet. Run the agent first.\n")
        return 1

    passed = sum(1 for r in ran if r["status"] == "PASS")
    clean = sum(1 for r in ran if r["status"] == "PASS" and r["path_ok"])

    print("  " + "-" * 68)
    print(f"  verdict correct : {passed}/{len(ran)}")
    print(f"  AND right route : {clean}/{len(ran)}")
    models = sorted({r["model"] for r in ran})
    if len(models) > 1:
        print(f"  NOTE: mixed models in this table ({', '.join(models)}).")
        print("        Re-run all three with one model before comparing.")

    for r in ran:
        if r["status"] == "PASS" and not r["path_ok"]:
            print(f"\n  Case {r['case']}: right answer, wrong route.")
            print(f"    never read: {', '.join(r['missing'])}")
            print(f"    read instead: {', '.join(r['extra']) or '(nothing extra)'}")
            print("    A correct verdict reached without the deciding record is")
            print("    luck, not reasoning. It should not count as a pass.")

    print()
    return 0 if clean == len(ran) else 1


if __name__ == "__main__":
    raise SystemExit(main())
