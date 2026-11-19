"""Запуск окна или команд в терминале. Оба варианта используют одну проверку документа."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from subprocess import SubprocessError

from sirius_proj.config import default_data_dir, default_settings
from sirius_proj.domain.models import CheckFailure


def main() -> int:
    parser = argparse.ArgumentParser(description="Локальная проверка технической документации")
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir(), help="Каталог компонентов приложения"
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("install", help="Подготовить модели и недостающие компоненты")
    check = commands.add_parser("check", help="Проверить документ без графического интерфейса")
    check.add_argument("document", type=Path)
    check.add_argument("checklist", type=Path)
    check.add_argument("-o", "--output", required=True, type=Path)
    check.add_argument("--json", type=Path, help="Дополнительно сохранить машинный отчёт")
    check.add_argument("--cpu", action="store_true")
    check.add_argument(
        "--rag", action="store_true", help="Использовать поиск даже для короткого отчёта"
    )
    args = parser.parse_args()
    settings = default_settings(args.data_dir)
    if args.command == "install":
        from sirius_proj.installation import install_components

        try:
            install_components(args.data_dir)
            return 0
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Не удалось подготовить компоненты: {error}", file=sys.stderr)
            return 1
    if args.command is None:
        from sirius_proj.ui.startup import prepare_qt

        try:
            prepare_qt(args.data_dir)
        except (OSError, RuntimeError) as error:
            print(str(error), file=sys.stderr)
            return 1
        from sirius_proj.bootstrap import launch_gui

        return launch_gui(settings)
    from sirius_proj.bootstrap import create_service, write_report

    try:
        report = create_service(replace(settings, use_gpu=not args.cpu, force_rag=args.rag)).run(
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
    except (ValueError, RuntimeError, OSError, SubprocessError) as error:
        print(f"Проверка не выполнена: {error}", file=sys.stderr)
        return 1


def report_to_dict(report) -> dict:
    from dataclasses import asdict

    return {
        "document": str(report.document.path),
        "seconds": report.seconds,
        "warnings": list(report.document.warnings),
        "cancelled": report.cancelled,
        "context_mode": report.context_mode,
        "timings": dict(report.timings),
        "results": [asdict(result) for result in report.results],
    }
