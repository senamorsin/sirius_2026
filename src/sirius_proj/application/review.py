"""Проверка документа целиком. Функции чтения, поиска и оценки передаются при создании."""

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from sirius_proj.application.checks import check_requirements
from sirius_proj.application.ports import Judge, SearchIndex
from sirius_proj.domain.models import (
    Block,
    CheckKind,
    CheckReport,
    CheckResult,
    DocumentText,
    Requirement,
)

FULL_CONTEXT_LIMIT = 15_000


@dataclass(frozen=True)
class FullContext:
    blocks: tuple[Block, ...]

    def retrieve(self, requirement: str) -> tuple[Block, ...]:
        return self.blocks


@dataclass(frozen=True)
class ReviewService:
    read_document: Callable[[Path], DocumentText]
    read_checklist: Callable[[Path], tuple[Requirement, ...]]
    build_index: Callable[[tuple[Block, ...], Callable[[], bool]], SearchIndex]
    open_judge: Callable[[Callable[[], bool]], AbstractContextManager[Judge]]

    force_rag: bool = False

    def run(
        self,
        document_path: Path,
        checklist_path: Path,
        *,
        on_result: Callable[[CheckResult], None] = lambda result: None,
        on_progress: Callable[[int, int, str], None] = lambda done, total, text: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> CheckReport:
        """Читает файлы, строит поиск, проверяет требования и собирает отчёт."""
        started = perf_counter()
        on_progress(0, 0, "Чтение документа и чек-листа…")
        document = self.read_document(document_path)
        requirements = self.read_checklist(checklist_path)
        read_at = perf_counter()
        if cancelled():
            raise InterruptedError("Проверка отменена")
        needs_model = any(req.kind is CheckKind.CONTENT for req in requirements)
        use_rag = (
            self.force_rag
            or sum(len(block.text) + 1 for block in document.blocks) >= FULL_CONTEXT_LIMIT
        )
        index = judge = None
        mode = "layout"
        with ExitStack() as resources:
            if needs_model and use_rag:
                mode = "rag"
                on_progress(0, 0, "Режим RAG. Индексация документа…")
                index = self.build_index(document.blocks, cancelled)
            elif needs_model:
                mode = "full"
                index = FullContext(document.blocks)
            indexed_at = perf_counter()
            if cancelled():
                raise InterruptedError("Проверка отменена")
            if needs_model:
                label = "RAG" if use_rag else "Весь документ"
                on_progress(0, 0, f"Режим «{label}». Загрузка языковой модели…")
                judge = resources.enter_context(self.open_judge(cancelled))
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
            checklist_path=checklist_path,
            context_mode=mode,
            timings=(
                ("read", read_at - started),
                ("index", indexed_at - read_at),
                ("model_load", model_at - indexed_at),
                ("checks", report.seconds),
            ),
        )
