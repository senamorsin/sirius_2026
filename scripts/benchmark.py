"""Reproducible synthetic 500-page benchmark. Not a real contractor report."""

import argparse
import csv
import json
import textwrap
from dataclasses import replace
from pathlib import Path

from create_examples import register_fonts
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from sirius_proj.application import run_job
from sirius_proj.cli import report_to_dict
from sirius_proj.domain import CheckFailure
from sirius_proj.export import export_xlsx
from sirius_proj.settings import default_settings

ROOT = Path(__file__).resolve().parents[1]


def make_document(path):
    register_fonts()
    pdf = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
    background = (
        "Служебное описание организации технических работ. Материал является синтетическим "
        "примером для измерения производительности приложения. Здесь описан порядок подготовки "
        "рабочего места, регистрации документа и размещения справочных материалов. "
        "Специалист сверяет записи с исходными сведениями и сохраняет результаты проверки. "
    ) * 7
    for page in range(1, 501):
        pdf.setFont("DejaVuBold", 13)
        pdf.drawString(50, 795, f"Учебный большой отчёт. Страница {page}")
        text = pdf.beginText(50, 765)
        text.setFont("DejaVu", 10)
        text.setLeading(14)
        for line in textwrap.wrap(background, width=88):
            text.textLine(line)
        text.textLine("")
        if page == 17:
            text.textLine("Напряжение питания шкафа БК-777 составляет 220 В.")
        if page == 251:
            text.textLine("Класс защиты корпуса шкафа БК-777: IP54.")
        pdf.drawText(text)
        pdf.showPage()
    pdf.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    folder = ROOT / "output/benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    document = folder / "500-pages.pdf"
    make_document(document)
    requirements = [
        ("B1", "Напряжение питания шкафа БК-777 должно составлять 220 В."),
        ("B2", "Класс защиты корпуса шкафа БК-777 должен быть IP65."),
        ("B3", "Отчёт должен содержать результаты испытания шкафа БК-777 на вибростойкость."),
    ]
    checklist = folder / "checklist.csv"
    with checklist.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "requirement"])
        writer.writerows(requirements)
    settings = default_settings(ROOT)
    if args.model:
        settings = replace(settings, model_file=args.model.resolve())
    report = run_job(
        document, checklist, settings, on_progress=lambda done, total, text: print(text, flush=True)
    )
    export_xlsx(
        folder / "results.xlsx", report.results, report.document.path, report.document.warnings
    )
    data = report_to_dict(report)
    data["document"] = "synthetic 500-page PDF"
    data["pages"] = 500
    data["characters"] = sum(len(block.text) for block in report.document.blocks)
    data["known_source_pages"] = {"B1": 17, "B2": 251}
    verified_sources = {}
    for result in report.results:
        rid = result.requirement.id
        if rid in data["known_source_pages"] and not isinstance(result, CheckFailure):
            verified_sources[rid] = bool(result.evidence) and all(
                item.location.page == data["known_source_pages"][rid] for item in result.evidence
            )
    data["all_citations_point_to_known_pages"] = (
        len(verified_sources) == len(data["known_source_pages"]) and all(verified_sources.values())
    )
    (folder / "metrics.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
