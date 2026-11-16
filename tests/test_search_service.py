"""Проверки поиска фрагментов, обработки требований и отмены."""

import threading
from pathlib import Path

import numpy as np
import pytest

from sirius_proj.adapters.llama_client import parse_answer
from sirius_proj.adapters.search import HybridIndex
from sirius_proj.application.checks import check_requirements
from sirius_proj.application.errors import CheckError
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
    index = HybridIndex(blocks(), Encoder(), limit=2)
    found = index.retrieve("Применён контроллер КТ-1234")
    assert found[0].id == "b1"
    assert len(found) == 2


def test_index_encodes_document_only_once():
    class CountingEncoder(Encoder):
        calls = []

        def encode(self, texts, *, query=False):
            self.calls.append(query)
            return super().encode(texts, query=query)

    encoder = CountingEncoder()
    index = HybridIndex(blocks(), encoder)
    index.retrieve("Первое")
    index.retrieve("Второе")
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
    index = HybridIndex(candidates, NearEncoder(), limit=1)
    assert index.retrieve("Сопротивление изоляции должно составлять 70 МОм.")[0].id == "b1"


class Index:
    def retrieve(self, requirement):
        return (Block("b1", "Питание 220 В", Location("pdf", page=1)),)


class Judge:
    def assess(self, requirement, chunks):
        return parse_answer(
            '{"status":"compliant","explanation":"Указано питание",'
            '"evidence":[{"chunk_id":"b1","quote":"Питание 220 В"}]}'
        )


def test_programming_error_is_not_presented_as_a_document_result():
    class BuggyIndex:
        def retrieve(self, text):
            raise RuntimeError("Внутренняя ошибка поиска")

    with pytest.raises(RuntimeError, match="Внутренняя ошибка поиска"):
        check_requirements(
            DocumentText(Path("report.pdf"), blocks()),
            (Requirement("1", "Питание"),),
            BuggyIndex(),
            Judge(),
        )


def test_service_reports_each_requirement_and_separates_failures():
    class BrokenJudge(Judge):
        def assess(self, requirement, chunks):
            if requirement.id == "2":
                raise CheckError("Модель не ответила")
            return super().assess(requirement, chunks)

    requirements = (Requirement("1", "Питание"), Requirement("2", "Мощность"))
    seen = []
    report = check_requirements(
        DocumentText(Path("report.pdf"), blocks()),
        requirements,
        Index(),
        BrokenJudge(),
        on_result=seen.append,
    )
    assert len(report.results) == 2
    assert report.results[0].status is Status.COMPLIANT
    assert isinstance(report.results[1], CheckFailure)
    assert "не ответила" in report.results[1].message
    assert seen == list(report.results)


def test_service_cancels_between_requirements():
    event = threading.Event()
    report = check_requirements(
        DocumentText(Path("report.pdf"), blocks()),
        (Requirement("1", "Питание"), Requirement("2", "Другое")),
        Index(),
        Judge(),
        on_result=lambda _: event.set(),
        cancelled=event.is_set,
    )
    assert report.cancelled
    assert len(report.results) == 1


def test_model_cannot_cite_unretrieved_document_block():
    class OtherIndex(Index):
        def retrieve(self, requirement):
            return (Block("b2", "Нет данных", Location("pdf", page=2)),)

    report = check_requirements(
        DocumentText(Path("report.pdf"), blocks()),
        (Requirement("1", "Питание"),),
        OtherIndex(),
        Judge(),
    )
    assert report.results[0].status is Status.INSUFFICIENT


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
    found = HybridIndex(chunks, SimilarEncoder(), limit=3).retrieve(
        "В ТК-101 должны быть подключены 16 дискретных входов."
    )
    assert found[0].id == "inputs"
