"""Таблица результатов и показ цитат. Связь строк с результатами хранится в самой таблице."""

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from sirius_proj.domain.models import CheckFailure, CheckResult
from sirius_proj.ui.style import COLORS


class ResultTable(QTableWidget):
    def __init__(self):
        super().__init__(0, 4)
        self._results: list[CheckResult] = []

    def clear_results(self):
        self.clearSelection()
        self._results.clear()
        self.setRowCount(0)

    def append_result(self, result: CheckResult):
        row = self.rowCount()
        self._results.append(result)
        self.insertRow(row)
        failure = isinstance(result, CheckFailure)
        status = "Техническая ошибка" if failure else result.status.label
        explanation = result.message if failure else result.explanation
        values = [result.requirement.id, result.requirement.text, status, explanation]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 2:
                item.setForeground(QColor("#b23a3a" if failure else COLORS[result.status]))
            self.setItem(row, column, item)
        self.resizeRowToContents(row)
        if row == 0:
            self.selectRow(0)

    def filter_status(self, selected: str):
        for row, result in enumerate(self._results):
            value = "error" if isinstance(result, CheckFailure) else result.status.value
            self.setRowHidden(row, selected not in ("all", value))

    def selected_result(self) -> CheckResult | None:
        row = self.currentRow()
        return self._results[row] if 0 <= row < len(self._results) else None


def describe_result(result: CheckResult) -> str:
    if isinstance(result, CheckFailure):
        return "Техническая ошибка\n\n" + result.message
    parts = [result.requirement.text, result.status.label, result.explanation]
    for number, item in enumerate(result.evidence, 1):
        parts.append(f"{number}. {item.location.label}\nФрагмент: {item.chunk_id}\n«{item.quote}»")
    if not result.evidence:
        parts.append("Подтверждённых цитат нет.")
    return "\n\n".join(parts)
