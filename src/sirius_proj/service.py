"""Application orchestration; independent of Qt and concrete adapters."""

from collections.abc import Callable, Sequence
from time import perf_counter

from .domain import CheckFailure, CheckReport, CheckResult, DocumentText, Requirement
from .evidence import verify_answer
from .ports import Judge, SearchIndex


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
        except (ValueError, RuntimeError, OSError, TimeoutError) as error:
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
