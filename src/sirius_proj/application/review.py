"""The complete use case. All I/O is supplied through its constructor."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from sirius_proj.application.checks import check_requirements
from sirius_proj.application.ports import Judge, SearchIndex
from sirius_proj.domain.models import Block, CheckReport, CheckResult, DocumentText, Requirement


@dataclass(frozen=True)
class ReviewService:
    read_document: Callable[[Path], DocumentText]
    read_checklist: Callable[[Path], tuple[Requirement, ...]]
    build_index: Callable[[tuple[Block, ...], Callable[[], bool]], SearchIndex]
    open_judge: Callable[[Callable[[], bool]], AbstractContextManager[Judge]]

    def run(
        self,
        document_path: Path,
        checklist_path: Path,
        *,
        on_result: Callable[[CheckResult], None] = lambda result: None,
        on_progress: Callable[[int, int, str], None] = lambda done, total, text: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> CheckReport:
        started = perf_counter()
        on_progress(0, 0, "Чтение документа и чек-листа…")
        document = self.read_document(document_path)
        requirements = self.read_checklist(checklist_path)
        read_at = perf_counter()
        if cancelled():
            raise InterruptedError("Проверка отменена")
        on_progress(0, 0, "Индексация документа…")
        index = self.build_index(document.blocks, cancelled)
        indexed_at = perf_counter()
        if cancelled():
            raise InterruptedError("Проверка отменена")
        on_progress(0, 0, "Загрузка языковой модели…")
        with self.open_judge(cancelled) as judge:
            model_at = perf_counter()
            report = check_requirements(
                document,
                requirements,
                index,
                judge,
                on_result=on_result,
                on_progress=on_progress,
                cancelled=cancelled,
            )
        return replace(
            report,
            seconds=perf_counter() - started,
            timings=(
                ("read", read_at - started),
                ("index", indexed_at - read_at),
                ("model_load", model_at - indexed_at),
                ("checks", report.seconds),
            ),
        )
