#!/usr/bin/env python3
"""Score results.json against the 19-task eval set.

Auto-grades: math, classification, short QA, extraction, reasoning (exact-ish match)
Exec-grades: code (runs the function against test cases)
Manual:      summarization, NER (prints for human review)

Usage:
    python3 eval/score.py eval/output/results.json
"""
import json
import re
import sys
from pathlib import Path

EXACT = {
    "math_01": ["408"],
    "math_02": ["80"],
    "math_03": ["15"],
    "cls_01": ["negative"],
    "cls_02": ["neutral"],
    "qa_01": ["canberra"],
    "qa_02": ["6", "six"],
    "qa_03": ["au"],
    "ext_01": ["help@acme-corp.io"],
    "ext_02": ["1919"],
    "rsn_01": ["no"],
    "rsn_02": ["ana"],
}

CODE_TESTS = {
    "code_01": (
        "is_palindrome",
        [(("A man, a plan, a canal: Panama",), True), (("hello",), False), (("",), True)],
    ),
    "code_02": (
        "fizzbuzz",
        [((5,), ["1", "2", "Fizz", "4", "Buzz"])],
    ),
    "code_03": (
        "average",
        [(([1, 2, 3],), 2.0), (([10, 20],), 15.0)],
    ),
}

MANUAL = ["summ_01", "summ_02", "ner_01", "ner_02"]


def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s`*_]+", " ", s)
    return s.strip(" .!?,;:'\"")


def grade_exact(tid: str, answer: str) -> bool:
    got = norm(answer)
    for want in EXACT[tid]:
        if got == want:
            return True
        # tolerate a short answer embedded in a one-liner
        if len(got) < 60 and re.search(rf"\b{re.escape(want)}\b", got):
            return True
    return False


def grade_code(tid: str, answer: str) -> bool:
    fname, cases = CODE_TESTS[tid]
    code = answer
    m = re.search(r"```(?:python)?\s*(.*?)```", answer, re.S)
    if m:
        code = m.group(1)
    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 - grading sandbox, local only
        fn = ns.get(fname)
        if fn is None:
            return False
        for args, want in cases:
            if fn(*args) != want:
                return False
        return True
    except Exception:
        return False


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/output/results.json")
    results = {r["task_id"]: (r.get("answer") or "") for r in json.loads(path.read_text())}

    auto_correct = 0
    auto_total = 0
    lines = []

    for tid in sorted(results):
        ans = results[tid]
        if tid in EXACT:
            ok = grade_exact(tid, ans)
            auto_total += 1
            auto_correct += ok
            lines.append(f"{'PASS' if ok else 'FAIL'}  {tid:9} -> {ans[:70]!r}")
        elif tid in CODE_TESTS:
            ok = grade_code(tid, ans)
            auto_total += 1
            auto_correct += ok
            lines.append(f"{'PASS' if ok else 'FAIL'}  {tid:9} -> (exec) {ans[:50]!r}")
        elif tid in MANUAL:
            lines.append(f"MANUAL {tid:9} -> {ans[:200]!r}")
        else:
            lines.append(f"????  {tid:9} unknown task id")

    print("\n".join(lines))
    print()
    print(f"AUTO-GRADED: {auto_correct}/{auto_total}")
    print(f"MANUAL REVIEW NEEDED: {len(MANUAL)} tasks (summarization + NER)")
    print()
    n_manual_ok = auto_total  # placeholder marker
    print("Gate math: you need 16/19 total (80%).")
    print(f"  Auto correct:        {auto_correct}")
    print(f"  Manual tasks:        {len(MANUAL)}  <- grade these by eye above")
    print(f"  Best case if all manual pass: {auto_correct + len(MANUAL)}/19")
    if auto_correct + len(MANUAL) < 16:
        print("  ** CANNOT CLEAR GATE even with perfect manual scores **")


if __name__ == "__main__":
    main()
