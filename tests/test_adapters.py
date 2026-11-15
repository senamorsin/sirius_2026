from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import sirius_proj.adapters.embeddings as embeddings
import sirius_proj.bootstrap as application
from sirius_proj import cli
from sirius_proj.adapters.llama_client import parse_answer
from sirius_proj.application.review import ReviewService
from sirius_proj.domain.models import (
    Block,
    CheckFailure,
    CheckReport,
    DocumentText,
    Location,
    Requirement,
)


@pytest.fixture
def encoder(tmp_path, monkeypatch):
    vocabulary = {"[UNK]": 0, "<pad>": 1, "word": 2, "query": 3, "passage": 4, ":": 5}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "model.onnx").touch()

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

        def run(self, outputs, feeds):
            ids = feeds["input_ids"]
            return [np.stack([ids, np.ones_like(ids)], axis=-1).astype(np.float32)]

    monkeypatch.setattr(embeddings.ort, "InferenceSession", lambda *args, **kwargs: Session())
    return embeddings.E5Encoder(tmp_path)


def test_e5_masked_pooling_and_query_prefix(encoder):
    result = encoder.encode(["word", "word word"])
    assert result.shape == (2, 2)
    assert np.allclose(np.linalg.norm(result, axis=1), 1)
    expected = np.array([11 / 3, 1])  # passage=4, colon=5, word=2
    expected /= np.linalg.norm(expected)
    assert np.allclose(result[0], expected)
    query = encoder.encode(["word"], query=True)
    assert not np.allclose(query, result[:1])


def test_chunk_offsets_overlap_and_source_survive(encoder):
    text = "word " * 900
    location = Location("pdf", page=7)
    chunks = encoder.split([Block("p7", text, location)])
    assert len(chunks) == 3
    assert all(chunk.location == location for chunk in chunks)
    assert chunks[0].id == "p7:c1"
    assert all(chunk.text in text for chunk in chunks)
    assert len(chunks[0].text.split()) == 400
    assert len(chunks[-1].text.split()) == 200
    assert encoder.split([Block("small", "word", location)])[0].id == "small"
    with pytest.raises(InterruptedError):
        encoder.split([Block("p7", text, location)], lambda: True)


def test_missing_embedding_weights_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="sirius-proj install"):
        embeddings.E5Encoder(tmp_path)


def test_use_case_accepts_injected_adapters_and_closes_judge():
    block = Block("b1", "220 В", Location("pdf", page=1))
    document = DocumentText(Path("doc.pdf"), (block,))

    class Judge:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def assess(self, requirement, chunks):
            return parse_answer(
                '{"status":"compliant","explanation":"Указано",'
                '"evidence":[{"chunk_id":"b1","quote":"220 В"}]}'
            )

    judge = Judge()
    service = ReviewService(
        lambda path: document,
        lambda path: (Requirement("1", "Питание"),),
        lambda blocks, cancelled: SimpleNamespace(retrieve=lambda text: blocks),
        lambda cancelled: judge,
    )
    report = service.run(Path("doc.pdf"), Path("list.csv"))
    assert judge.closed
    assert report.results[0].evidence[0].quote == "220 В"
    assert dict(report.timings)["read"] >= 0
    with pytest.raises(InterruptedError):
        service.run(Path("doc.pdf"), Path("list.csv"), cancelled=lambda: True)


def test_cli_export_json_and_technical_exit_code(tmp_path, monkeypatch):
    result = CheckFailure(Requirement("1", "Питание"), "Модель недоступна")
    report = CheckReport(DocumentText(Path("doc.pdf"), ()), (result,), 0.2)
    monkeypatch.setattr(
        application,
        "create_service",
        lambda settings: SimpleNamespace(run=lambda *args, **kwargs: report),
    )
    output = tmp_path / "report.xlsx"
    raw = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sirius-proj",
            "check",
            "doc.pdf",
            "list.csv",
            "-o",
            str(output),
            "--json",
            str(raw),
            "--cpu",
        ],
    )
    assert cli.main() == 2
    assert output.is_file()
    assert "Модель недоступна" in raw.read_text()

    def fail(*args, **kwargs):
        raise ValueError("Некорректный CSV")

    monkeypatch.setattr(application, "create_service", fail)
    assert cli.main() == 1
