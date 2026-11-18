"""Проверки свойств Word и структуры отчёта по извлечённым данным."""

import re

from sirius_proj.domain.models import (
    Block,
    CheckKind,
    DocumentText,
    DocxLayout,
    Evidence,
    Finding,
    Location,
    Requirement,
    Status,
)


def check_layout(requirement: Requirement, document: DocumentText) -> Finding:
    layout = document.layout
    text = requirement.text.casefold()
    if layout is None:
        return _unknown(requirement, "Для проверки свойств и структуры нужен исходный файл DOCX.")
    if requirement.kind is CheckKind.METADATA and re.search(r"\b[аa]4\b", text):
        return _page_size(requirement, layout)
    if requirement.kind is CheckKind.STRUCTURE:
        if "нумерац" in text and ("таблиц" in text or "рисунк" in text):
            return _numbering(requirement, layout)
        match = re.search(r'[«"“]([^»"”]+)[»"”]', requirement.text)
        if "раздел" in text and match:
            return _section(requirement, document, layout, match[1])
    return _unknown(requirement, "Для этого требования нет программного правила проверки.")


def _unknown(requirement: Requirement, explanation: str) -> Finding:
    return Finding(requirement, Status.INSUFFICIENT, explanation, ())


def _observation(key: str, source: str, text: str) -> Evidence:
    return Evidence(key, text, Location("layout", heading=source))


def _page_size(requirement: Requirement, layout: DocxLayout) -> Finding:
    known = [sorted(size) for size in layout.page_sizes if size is not None]
    wrong = any(abs(width - 210) > 1 or abs(height - 297) > 1 for width, height in known)
    evidence = tuple(
        _observation(
            f"section{i}",
            f"Параметры страницы DOCX, секция {i}",
            f"{size[0]:.2f} × {size[1]:.2f} мм" if size else "Размер страницы не указан",
        )
        for i, size in enumerate(layout.page_sizes, 1)
    )
    if wrong:
        status, reason = Status.NON_COMPLIANT, "Размер хотя бы одной секции отличается от A4."
    elif known and len(known) == len(layout.page_sizes):
        status, reason = Status.COMPLIANT, "Размер страниц всех секций соответствует A4."
    else:
        status, reason = Status.INSUFFICIENT, "Размер страниц указан не во всех секциях."
    return Finding(requirement, status, reason, evidence)


def _numbering(requirement: Requirement, layout: DocxLayout) -> Finding:
    text = requirement.text.casefold()
    groups = []
    if "таблиц" in text:
        groups.append(("Таблицы", layout.table_numbers))
    if "рисунк" in text:
        groups.append(("Рисунки", layout.figure_numbers))
    present = [numbers for _, numbers in groups if numbers]
    if not present:
        return _unknown(requirement, "В основном тексте нет указанных объектов для проверки.")
    if not any(any(numbers) for numbers in present):
        status, reason = Status.NON_COMPLIANT, "Номера у найденных таблиц и рисунков отсутствуют."
    elif all(list(numbers) == [str(i) for i in range(1, len(numbers) + 1)] for numbers in present):
        status, reason = Status.COMPLIANT, "Номера идут подряд с единицы для каждого вида объектов."
    else:
        status, reason = (
            Status.PARTIAL,
            "Номера есть, но последовательность неполная или не сквозная.",
        )
    evidence = tuple(
        _observation(
            f"numbering{i}",
            "Подписи в основном тексте DOCX",
            f"{label}: " + (", ".join(n or "без номера" for n in numbers) or "отсутствуют"),
        )
        for i, (label, numbers) in enumerate(groups, 1)
    )
    return Finding(requirement, status, reason, evidence)


def _heading(text: str) -> str:
    return " ".join(re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", text).casefold().split())


def _body(document: DocumentText, layout: DocxLayout, block: Block) -> tuple[Block, ...]:
    next_heading = next(
        (h for h in layout.headings if h.location.paragraph > block.location.paragraph), None
    )
    position = document.blocks.index(block)
    end = document.blocks.index(next_heading) if next_heading else len(document.blocks)
    return document.blocks[position + 1 : end]


def _cite(block: Block) -> Evidence:
    return Evidence(block.id, block.text, block.location)


def _section(
    requirement: Requirement, document: DocumentText, layout: DocxLayout, title: str
) -> Finding:
    wanted = _heading(title)
    headings = sorted(layout.headings, key=lambda h: _heading(h.text) != wanted)
    for heading in headings:
        actual = _heading(heading.text)
        conclusion = wanted == "заключение" and "заключение" in actual.split()
        definitions = wanted == "список сокращений и определений" and (
            "определен" in actual or "сокращен" in actual
        )
        if actual != wanted and not conclusion and not definitions:
            continue
        body = _body(document, layout, heading)
        if definitions:
            return _definitions(requirement, heading, body, actual == wanted)
        if conclusion and not body:
            return Finding(
                requirement, Status.PARTIAL, "Есть заголовок, но раздел пуст.", (_cite(heading),)
            )
        evidence = (_cite(heading), *(_cite(block) for block in body[:1] if conclusion))
        return Finding(
            requirement,
            Status.COMPLIANT,
            f"Найден самостоятельный раздел «{heading.text}».",
            evidence,
        )
    if not headings:
        return _unknown(
            requirement, "В документе нет размеченных заголовков для проверки разделов."
        )
    listing = "; ".join(h.text for h in layout.headings)
    evidence = _observation("headings", "Заголовки основного текста DOCX", listing)
    return Finding(
        requirement, Status.NON_COMPLIANT, f"Отдельного раздела «{title}» нет.", (evidence,)
    )


def _definitions(
    requirement: Requirement, heading: Block, body: tuple[Block, ...], exact: bool
) -> Finding:
    abbreviations, definitions = [], []
    for block in body:
        for term, _ in re.findall(r"(?m)^([^\n]+?)\s+[–—-]\s+([^\n]+)", block.text):
            (abbreviations if term.isupper() else definitions).append(block)
    if not exact and not (abbreviations or definitions):
        return _unknown(requirement, "Точный заголовок и список с определениями не найдены.")
    complete = exact and abbreviations and definitions
    evidence = tuple(
        _cite(block) for block in dict.fromkeys([heading, *abbreviations[:1], *definitions[:1]])
    )
    fields = []
    if not exact:
        fields.append("название раздела отличается")
    if not abbreviations or not definitions:
        fields.append("сокращения и определения представлены не полностью")
    reason = "Найдены сокращения и определения в разделе с требуемым названием."
    if fields:
        reason = "Раздел найден, но " + ", ".join(fields) + "."
    return Finding(requirement, Status.COMPLIANT if complete else Status.PARTIAL, reason, evidence)
