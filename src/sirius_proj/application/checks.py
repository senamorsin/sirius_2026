"""Проверка требований по очереди. Работает независимо от окна и выбранных библиотек."""

from collections.abc import Callable, Sequence

from sirius_proj.application.errors import CheckError
from sirius_proj.application.ports import Judge
from sirius_proj.application.structure import check_layout
from sirius_proj.domain.evidence import verify_answer
from sirius_proj.domain.models import (
    Block,
    CheckFailure,
    CheckKind,
    CheckResult,
    DocumentText,
    Finding,
    Requirement,
)


def check_requirement(
    document: DocumentText,
    retrieve: Callable[[str], tuple[Block, ...]],
    judge: Judge,
    requirement: Requirement,
) -> Finding:
    if requirement.kind is not CheckKind.CONTENT:
        return check_layout(requirement, document)
    query = "\n".join(filter(None, (requirement.text, requirement.comment)))
    chunks = retrieve(query)
    answer = judge.assess(requirement, chunks)
    return verify_answer(requirement, answer, {chunk.id: chunk for chunk in chunks})


def check_requirements(
    requirements: Sequence[Requirement],
    check: Callable[[Requirement], Finding],
    *,
    on_result: Callable[[CheckResult], None] = lambda result: None,
    on_progress: Callable[[int, int, str], None] = lambda done, total, message: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[CheckResult, ...]:
    """Проверяет требования по одному и сообщает о каждом готовом результате."""
    results = []
    for number, requirement in enumerate(requirements, 1):
        if cancelled():
            break
        on_progress(number - 1, len(requirements), f"Проверка {requirement.id}: {requirement.text}")
        try:
            result = check(requirement)
        except CheckError as error:
            result = CheckFailure(requirement, str(error))
        results.append(result)
        on_result(result)
        on_progress(number, len(requirements), f"Обработано {number} из {len(requirements)}")
    return tuple(results)
