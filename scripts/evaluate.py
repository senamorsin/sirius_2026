"""Evaluate only the public synthetic examples; never manufacture curator validation."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from sirius_proj.application import run_job
from sirius_proj.cli import report_to_dict
from sirius_proj.domain import CheckFailure
from sirius_proj.evidence import normalize
from sirius_proj.export import export_xlsx
from sirius_proj.settings import default_settings

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["docx", "pdf"], choices=["docx", "pdf"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/evaluation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = replace(default_settings(ROOT), use_gpu=not args.cpu)
    if args.model:
        settings = replace(settings, model_file=args.model.resolve())
    expected = json.loads((ROOT / "examples/expected.json").read_text())
    summary = {
        "dataset": "public synthetic examples; not curator validation",
        "model": str(settings.model_file.relative_to(ROOT)),
        "cpu": args.cpu,
        "runs": {},
    }
    for extension in args.formats:
        report = run_job(
            ROOT / f"examples/report.{extension}",
            ROOT / "examples/checklist.csv",
            settings,
            on_progress=lambda done, total, text: print(text, flush=True),
        )
        export_xlsx(
            args.output_dir / f"{extension}.xlsx",
            report.results,
            report.document.path,
            report.document.warnings,
        )
        raw = report_to_dict(report)
        raw["document"] = f"examples/report.{extension}"
        (args.output_dir / f"{extension}.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2)
        )
        matches = sum(
            not isinstance(item, CheckFailure)
            and item.status.value == expected[item.requirement.id]
            for item in report.results
        )
        citations = [
            e
            for item in report.results
            if not isinstance(item, CheckFailure)
            for e in item.evidence
        ]
        verified = sum(
            any(
                e.location == b.location and normalize(e.quote) in normalize(b.text)
                for b in report.document.blocks
            )
            for e in citations
        )
        summary["runs"][extension] = {
            "matched": matches,
            "total": len(expected),
            "seconds": report.seconds,
            "quotes": len(citations),
            "verified_quotes": verified,
            "technical_errors": sum(isinstance(item, CheckFailure) for item in report.results),
        }
        print(extension, summary["runs"][extension], flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
