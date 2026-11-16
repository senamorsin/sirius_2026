"""Проверка требований по очереди. Работает независимо от окна и выбранных библиотек."""

from collections.abc import Callable, Sequence
from time import perf_counter

from sirius_proj.application.errors import CheckError
from sirius_proj.application.ports import Judge, SearchIndex
from sirius_proj.domain.evidence import verify_answer
from sirius_proj.domain.models import (
    CheckFailure,
    CheckReport,
    CheckResult,
    DocumentText,
    Requirement,
)


def check_requirements(
    document: DocumentText,
    requirements: Sequence[Requirement],
    index: SearchIndex,
    judge: Judge,
    *,
    on_result: Callable[[CheckResult], None] = lambda result: None,
    on_progress: Callable[[int, int, str], None] = lambda done, total, message: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> CheckReport:
    """Проверяет требования по одному и сообщает о каждом готовом результате."""
    started = perf_counter()
    results = []
    for number, requirement in enumerate(requirements, 1):
        if cancelled():
            break
        on_progress(number - 1, len(requirements), f"Проверка {requirement.id}: {requirement.text}")
        try:
            chunks = index.retrieve(requirement.text)
            answer = judge.assess(requirement, chunks)
            result = verify_answer(requirement, answer, {chunk.id: chunk for chunk in chunks})
        except CheckError as error:
            result = CheckFailure(requirement, str(error))
        results.append(result)
        on_result(result)
        on_progress(number, len(requirements), f"Обработано {number} из {len(requirements)}")
    return CheckReport(
        document,
        tuple(results),
        perf_counter() - started,
        cancelled() and len(results) < len(requirements),
    )
