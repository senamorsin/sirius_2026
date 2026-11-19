"""Проверки поиска фрагментов, обработки требований и отмены."""

import threading
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from sirius_proj.adapters.llama_client import parse_answer
from sirius_proj.adapters.search import HybridIndex
from sirius_proj.application.checks import check_requirement, check_requirements
from sirius_proj.application.errors import CheckError
from sirius_proj.application.review import ReviewService
from sirius_proj.domain.models import (
    Block,
    CheckFailure,
    DocumentText,
    Location,
    Requirement,
    Status,
)


class Encoder:
    def encode(self, texts, *, query=False):
        if query:
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)
        return np.array([[1.0, 0.0], [0.1, 0.99], [0.9, 0.1]], dtype=np.float32)


def blocks():
    return tuple(
        Block(f"b{i}", text, Location("pdf", page=i + 1))
        for i, text in enumerate(
            [
                "Общие сведения о тяговом оборудовании",
                "Контроллер КТ-1234 снят с производства",
                "Оборудование проверено",
            ]
        )
    )


def test_exact_identifier_is_not_lost_to_semantic_ranking():
    index = HybridIndex(tuple(block.text for block in blocks()), Encoder())
    assert index.rank("Применён контроллер КТ-1234")[0] == 1


def test_index_encodes_document_only_once():
    class CountingEncoder(Encoder):
        calls = []

        def encode(self, texts, *, query=False):
            self.calls.append(query)
            return super().encode(texts, query=query)

    encoder = CountingEncoder()
    index = HybridIndex(tuple(block.text for block in blocks()), encoder)
    index.rank("Первое")
    index.rank("Второе")
    assert encoder.calls == [False, True, True]


def test_contents_entry_does_not_displace_actual_parameter():
    class NearEncoder(Encoder):
        def encode(self, texts, *, query=False):
            if query:
                return super().encode(texts, query=True)
            return np.array([[1.0, 0.0], [0.96, 0.28], [0.9, 0.1]], dtype=np.float32)

    candidates = tuple(
        Block(f"b{i}", text, Location("docx", paragraph=i + 1))
        for i, text in enumerate(
            [
                "Сопротивление изоляции .......................... 8",
                "Сопротивление изоляции составляет 70 МОм.",
                "Прочие сведения о документе",
            ]
        )
    )
    index = HybridIndex(tuple(block.text for block in candidates), NearEncoder())
    assert index.rank("Сопротивление изоляции должно составлять 70 МОм.")[0] == 1


class Index:
    def retrieve(self, requirement):
        return (Block("b1", "Питание 220 В", Location("pdf", page=1)),)


class Judge:
    def assess(self, requirement, chunks):
        return parse_answer(
            '{"status":"compliant","explanation":"Указано питание",'
            '"evidence":[{"chunk_id":"b1","quote":"Питание 220 В"}]}'
        )


def item_check(index, judge):
    return partial(
        check_requirement, DocumentText(Path("report.pdf"), blocks()), index.retrieve, judge
    )


def test_programming_error_is_not_presented_as_a_document_result():
    class BuggyIndex:
        def retrieve(self, text):
            raise RuntimeError("Внутренняя ошибка поиска")

    with pytest.raises(RuntimeError, match="Внутренняя ошибка поиска"):
        check_requirements((Requirement("1", "Питание"),), item_check(BuggyIndex(), Judge()))


def test_service_reports_each_requirement_and_separates_failures():
    class BrokenJudge(Judge):
        def assess(self, requirement, chunks):
            if requirement.id == "2":
                raise CheckError("Модель не ответила")
            return super().assess(requirement, chunks)

    requirements = (Requirement("1", "Питание"), Requirement("2", "Мощность"))
    seen = []
    results = check_requirements(
        requirements, item_check(Index(), BrokenJudge()), on_result=seen.append
    )
    assert len(results) == 2
    assert results[0].status is Status.COMPLIANT
    assert isinstance(results[1], CheckFailure)
    assert "не ответила" in results[1].message
    assert seen == list(results)


def test_service_cancels_between_requirements():
    event = threading.Event()
    service = ReviewService(
        lambda path: DocumentText(path, Index().retrieve("")),
        lambda _: (Requirement("1", "Питание"), Requirement("2", "Другое")),
        lambda *args: Index(),
        lambda _: nullcontext(Judge()),
    )
    report = service.run(
        Path("report.pdf"),
        Path("checklist.csv"),
        on_result=lambda _: event.set(),
        cancelled=event.is_set,
    )
    assert report.cancelled
    assert len(report.results) == 1


def test_model_cannot_cite_unretrieved_document_block():
    class OtherIndex(Index):
        def retrieve(self, requirement):
            return (Block("b2", "Нет данных", Location("pdf", page=2)),)

    result = item_check(OtherIndex(), Judge())(Requirement("1", "Питание"))
    assert result.status is Status.INSUFFICIENT


def test_frequent_equipment_identifier_does_not_hide_parameter_without_repeated_name():
    class SimilarEncoder:
        def encode(self, texts, *, query=False):
            if query:
                return np.array([[1.0, 0.0]], dtype=np.float32)
            return np.array([[0.8, 0.6]] * 10 + [[1.0, 0.0]], dtype=np.float32)

    chunks = tuple(
        Block(f"other{i}", "ТК-101: общие сведения о поставке.", Location("docx", paragraph=i + 1))
        for i in range(10)
    ) + (Block("inputs", "Подключено 16 дискретных входов.", Location("docx", paragraph=11)),)
    index = HybridIndex(tuple(block.text for block in chunks), SimilarEncoder())
    assert index.rank("В ТК-101 должны быть подключены 16 дискретных входов.")[0] == 10
