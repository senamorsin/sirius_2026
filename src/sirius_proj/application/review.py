"""Проверка документа целиком. Функции чтения, поиска и оценки передаются при создании."""

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter

from sirius_proj.application.checks import check_requirement, check_requirements
from sirius_proj.application.ports import Judge, SearchIndex
from sirius_proj.application.structure import check_layout
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
        check = partial(check_layout, document=document)
        mode = "layout"
        indexed_at = read_at
        with ExitStack() as resources:
            if any(req.kind is CheckKind.CONTENT for req in requirements):
                length = sum(len(block.text) + 1 for block in document.blocks)
                use_rag = self.force_rag or length >= FULL_CONTEXT_LIMIT
                mode = "rag" if use_rag else "full"
                if use_rag:
                    on_progress(0, 0, "Режим RAG. Индексация документа…")
                    retrieve = self.build_index(document.blocks, cancelled).retrieve
                else:

                    def retrieve(query: str) -> tuple[Block, ...]:
                        return document.blocks

                indexed_at = perf_counter()
                if cancelled():
                    raise InterruptedError("Проверка отменена")
                label = "RAG" if use_rag else "Весь документ"
                on_progress(0, 0, f"Режим «{label}». Загрузка языковой модели…")
                judge = resources.enter_context(self.open_judge(cancelled))
                check = partial(check_requirement, document, retrieve, judge)
            model_at = perf_counter()
            results = check_requirements(
                requirements,
                check,
                on_result=on_result,
                on_progress=on_progress,
                cancelled=cancelled,
            )
            checked_at = perf_counter()
        return CheckReport(
            document,
            results,
            seconds=perf_counter() - started,
            cancelled=len(results) < len(requirements),
            checklist_path=checklist_path,
            context_mode=mode,
            timings=(
                ("read", read_at - started),
                ("index", indexed_at - read_at),
                ("model_load", model_at - indexed_at),
                ("checks", checked_at - model_at),
            ),
        )
