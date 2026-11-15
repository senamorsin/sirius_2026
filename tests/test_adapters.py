from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from sirius_proj import application, cli, embeddings
from sirius_proj.domain import Block, CheckFailure, CheckReport, DocumentText, Location, Requirement
from sirius_proj.evidence import parse_answer
from sirius_proj.settings import Settings


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
    with pytest.raises(ValueError, match="prepare_models"):
        embeddings.E5Encoder(tmp_path)


def test_composition_uses_one_index_and_owned_judge(tmp_path, monkeypatch):
    settings = Settings(Path("server"), Path("model"), Path("e5"))
    monkeypatch.setattr(Settings, "validate", lambda self: None)
    block = Block("b1", "220 В", Location("pdf", page=1))
    document = DocumentText(Path("doc.pdf"), (block,))
    monkeypatch.setattr(application, "read_document", lambda path: document)
    monkeypatch.setattr(application, "load_checklist", lambda path: (Requirement("1", "Питание"),))
    monkeypatch.setattr(
        application, "E5Encoder", lambda folder: SimpleNamespace(split=lambda *args: (block,))
    )
    monkeypatch.setattr(
        application, "HybridIndex", lambda *args: SimpleNamespace(retrieve=lambda text: (block,))
    )

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
    monkeypatch.setattr(application, "LlamaRuntime", lambda *args: judge)
    report = application.run_job(Path("doc.pdf"), Path("list.csv"), settings)
    assert judge.closed
    assert report.results[0].evidence[0].quote == "220 В"
    assert dict(report.timings)["read"] >= 0
    with pytest.raises(InterruptedError):
        application.run_job(Path("doc.pdf"), Path("list.csv"), settings, cancelled=lambda: True)


def test_cli_export_json_and_technical_exit_code(tmp_path, monkeypatch):
    result = CheckFailure(Requirement("1", "Питание"), "Модель недоступна")
    report = CheckReport(DocumentText(Path("doc.pdf"), ()), (result,), 0.2)
    monkeypatch.setattr(application, "run_job", lambda *args, **kwargs: report)
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

    monkeypatch.setattr(application, "run_job", fail)
    assert cli.main() == 1
