"""GUI by default, the identical application scenario for reproducible CLI runs."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from sirius_proj.config import default_settings
from sirius_proj.domain.models import CheckFailure


def main() -> int:
    parser = argparse.ArgumentParser(description="Локальная проверка технической документации")
    parser.add_argument(
        "--data-dir", type=Path, default=Path.cwd(), help="Каталог проекта с models/ и runtime/"
    )
    commands = parser.add_subparsers(dest="command")
    check = commands.add_parser("check", help="Проверить документ без графического интерфейса")
    check.add_argument("document", type=Path)
    check.add_argument("checklist", type=Path)
    check.add_argument("-o", "--output", required=True, type=Path)
    check.add_argument("--json", type=Path, help="Дополнительно сохранить машинный отчёт")
    check.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    settings = default_settings(args.data_dir)
    if args.command is None:
        from .ui.startup import prepare_qt

        try:
            prepare_qt(args.data_dir)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 1
        from sirius_proj.bootstrap import launch_gui

        return launch_gui(settings)
    from sirius_proj.bootstrap import create_service, write_report

    try:
        report = create_service(replace(settings, use_gpu=not args.cpu)).run(
            args.document,
            args.checklist,
            on_progress=lambda done, total, text: print(text, flush=True),
        )
        write_report(args.output, report)
        if args.json:
            args.json.write_text(
                json.dumps(report_to_dict(report), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"Готово: {args.output} · {report.seconds:.2f} с")
        return 2 if any(isinstance(item, CheckFailure) for item in report.results) else 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Проверка не выполнена: {error}", file=sys.stderr)
        return 1


def report_to_dict(report) -> dict:
    from dataclasses import asdict

    return {
        "document": str(report.document.path),
        "seconds": report.seconds,
        "warnings": list(report.document.warnings),
        "cancelled": report.cancelled,
        "timings": dict(report.timings),
        "results": [asdict(result) for result in report.results],
    }
