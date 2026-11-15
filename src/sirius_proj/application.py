"""Composition root shared by GUI and headless CLI."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from .documents import load_checklist, read_document
from .domain import CheckReport, CheckResult
from .embeddings import E5Encoder
from .llama import LlamaRuntime
from .search import HybridIndex
from .service import check_requirements
from .settings import Settings


def run_job(
    document_path: Path,
    checklist_path: Path,
    settings: Settings,
    *,
    on_result: Callable[[CheckResult], None] = lambda result: None,
    on_progress: Callable[[int, int, str], None] = lambda done, total, message: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> CheckReport:
    started = perf_counter()
    on_progress(0, 0, "Чтение документа и чек-листа…")
    document = read_document(document_path)
    requirements = load_checklist(checklist_path)
    settings.validate()
    on_progress(0, 0, "Загрузка локальной модели эмбеддингов…")
    encoder = E5Encoder(settings.embedding_dir)
    chunks = encoder.split(document.blocks, cancelled)
    on_progress(0, 0, f"Индексация {len(chunks)} фрагментов…")
    index = HybridIndex(chunks, encoder)
    if cancelled():
        raise InterruptedError("Проверка отменена")
    on_progress(0, 0, "Загрузка языковой модели…")
    with LlamaRuntime(settings, cancelled) as judge:
        report = check_requirements(
            document,
            requirements,
            index,
            judge,
            on_result=on_result,
            on_progress=on_progress,
            cancelled=cancelled,
        )
    return replace(report, seconds=perf_counter() - started)
