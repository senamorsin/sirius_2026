"""Local substitutes for unavailable crapper/dryer: Radon CRAP and AST similarity.

CRAP uses measured statement coverage. AST comparison covers functions with >=10
statements, at threshold .8. Mutation testing uses the separate mutmut command.
"""

import ast
import difflib
import json
from itertools import combinations
from pathlib import Path

from radon.complexity import cc_visit
from radon.visitors import Class

ROOT = Path(__file__).resolve().parents[1]


def functions(blocks):
    seen = set()
    for block in blocks:
        candidates = block.methods if isinstance(block, Class) else [block]
        for candidate in candidates:
            key = (candidate.lineno, candidate.name)
            if key not in seen:
                seen.add(key)
                yield candidate


def crap_records(path, measured):
    executed = set(measured.get("executed_lines", []))
    missing = set(measured.get("missing_lines", []))
    for function in functions(cc_visit(path.read_text())):
        lines = (executed | missing) & set(range(function.lineno, function.endline + 1))
        fraction = len(executed & lines) / len(lines) if lines else 0.0
        complexity = function.complexity
        yield {
            "file": str(path.relative_to(ROOT)),
            "function": function.name,
            "line": function.lineno,
            "complexity": complexity,
            "coverage": fraction,
            "crap": complexity**2 * (1 - fraction) ** 3 + complexity,
        }


def ast_functions(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef):
            continue
        if sum(isinstance(child, ast.stmt) for child in ast.walk(node)) < 10:
            continue
        yield (
            f"{path.name}:{node.lineno}:{node.name}",
            ast.dump(node, include_attributes=False).split(", "),
        )


def main():
    measured = json.loads((ROOT / "coverage.json").read_text())["files"]
    records, chunks = [], []
    for path in sorted((ROOT / "src/sirius_proj").rglob("*.py")):
        records.extend(crap_records(path, measured.get(str(path.relative_to(ROOT)), {})))
        chunks.extend(ast_functions(path))
    duplicates = []
    for (first, a), (second, b) in combinations(chunks, 2):
        ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        if ratio >= 0.8:
            duplicates.append({"first": first, "second": second, "similarity": ratio})
    report = {
        "crap_formula": "complexity^2 * (1 - statement_coverage)^3 + complexity",
        "functions": records,
        "crap_violations": [r for r in records if r["crap"] >= 10],
        "ast_duplication_threshold": 0.8,
        "duplicate_candidates": duplicates,
    }
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output/quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"CRAP >= 10: {len(report['crap_violations'])}; AST similarity >= 0.8: {len(duplicates)}")
    raise SystemExit(bool(report["crap_violations"] or duplicates))


if __name__ == "__main__":
    main()
