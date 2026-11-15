"""Rebuild public synthetic fixtures; no curator or confidential data involved."""

import csv
import json
from pathlib import Path

from docx import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    ("Напряжение питания", "220 В"),
    ("Потребляемая мощность", "5 кВт"),
    ("Контроллер", "КТ-1234"),
    ("Диапазон температуры эксплуатации", "от -10 до +40 °C"),
    ("Класс защиты корпуса", "IP54"),
    ("Длина кабеля", "18 м"),
    ("Срок гарантии", "12 месяцев"),
    ("Редакция руководства по эксплуатации", "2"),
]
REQUIREMENTS = [
    ("C01", "Напряжение питания шкафа ТК-101 должно составлять 220 В.", "compliant"),
    ("C02", "Потребляемая мощность шкафа ТК-101 должна составлять 5 кВт.", "compliant"),
    ("C03", "В шкафу ТК-101 должен использоваться контроллер КТ-1234.", "compliant"),
    ("C04", "Диапазон температуры эксплуатации шкафа ТК-101: от -10 до +40 °C.", "compliant"),
    ("N01", "Класс защиты корпуса шкафа ТК-101 должен быть IP65.", "non_compliant"),
    ("N02", "Длина кабеля шкафа ТК-101 должна составлять 25 м.", "non_compliant"),
    ("N03", "Срок гарантии шкафа ТК-101 должен составлять 24 месяца.", "non_compliant"),
    ("N04", "Редакция руководства по эксплуатации шкафа ТК-101 должна быть 4.", "non_compliant"),
    (
        "U01",
        "Отчёт должен содержать измеренное сопротивление изоляции шкафа ТК-101.",
        "insufficient",
    ),
    (
        "U02",
        "Отчёт должен содержать дату последней калибровки измерительного прибора.",
        "insufficient",
    ),
    ("U03", "Отчёт должен содержать серийный номер установленного блока питания.", "insufficient"),
    (
        "U04",
        "Отчёт должен содержать результаты испытания шкафа ТК-101 на вибростойкость.",
        "insufficient",
    ),
]


def register_fonts():
    folder = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DejaVu", str(folder / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuBold", str(folder / "DejaVuSans-Bold.ttf")))


def main():
    folder = ROOT / "examples"
    folder.mkdir(exist_ok=True)
    title = "Учебный отчёт об испытаниях шкафа ТК-101"
    intro = "Синтетический пример для разработки. Не является документом Горэлектротранса."
    summary = (
        "Испытания шкафа ТК-101 выполнены 12 ноября 2026 года. "
        "Фактические параметры приведены в таблице."
    )
    document = Document()
    document.add_heading(title, 0)
    document.add_paragraph(intro)
    document.add_heading("Результаты испытаний", 1)
    document.add_paragraph(summary)
    table = document.add_table(rows=1, cols=2, style="Light Shading Accent 1")
    table.rows[0].cells[0].text = "Параметр"
    table.rows[0].cells[1].text = "Фактическое значение"
    for key, value in ROWS:
        row = table.add_row()
        row.cells[0].text, row.cells[1].text = key, value
    document.add_heading("Заключение", 1)
    document.add_paragraph("Визуальный осмотр выполнен. Повреждений корпуса не обнаружено.")
    document.save(folder / "report.docx")
    register_fonts()
    style = ParagraphStyle("body", fontName="DejaVu", fontSize=11, leading=17)
    heading = ParagraphStyle(
        "heading", parent=style, fontName="DejaVuBold", fontSize=16, leading=23
    )
    story = [
        Paragraph(title, heading),
        Spacer(1, 12),
        Paragraph(intro, style),
        Spacer(1, 12),
        Paragraph("Результаты испытаний", heading),
        Paragraph(summary, style),
        Spacer(1, 12),
    ]
    data = [
        [Paragraph(key, style), Paragraph(value, style)]
        for key, value in [("Параметр", "Фактическое значение"), *ROWS]
    ]
    pdf_table = Table(data, colWidths=[290, 190])
    pdf_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2eee9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story += [
        pdf_table,
        Spacer(1, 12),
        Paragraph("Заключение", heading),
        Paragraph("Визуальный осмотр выполнен. Повреждений корпуса не обнаружено.", style),
    ]
    SimpleDocTemplate(str(folder / "report.pdf"), pagesize=A4).build(story)
    with (folder / "checklist.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "requirement"])
        writer.writerows((rid, text) for rid, text, _ in REQUIREMENTS)
    (folder / "expected.json").write_text(
        json.dumps({rid: status for rid, _, status in REQUIREMENTS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (folder / "README.md").write_text(
        "# Учебные данные\n\nСинтетический отчёт в двух форматах и 12 требований: "
        "четыре соответствия, четыре явных противоречия, четыре случая недостатка данных.\n"
        "Эталон: expected.json. Пересоздание: uv run python scripts/create_examples.py.\n"
        "Это демонстрационный набор, он не измеряет качество на реальных отчётах подрядчиков.\n",
        encoding="utf-8",
    )
    print(folder)


if __name__ == "__main__":
    main()
