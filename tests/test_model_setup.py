import io
import os
import tarfile

from sirius_proj.installation import extract_runtime


def test_runtime_install_keeps_server_libraries_and_license_without_archive(tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        for name in ["llama-server", "libllama.so.1", "LICENSE", "llama-bench"]:
            item = tarfile.TarInfo("build/" + name)
            item.size = 4
            item.mode = 0o755
            package.addfile(item, io.BytesIO(b"test"))
    destination = tmp_path / "runtime"
    extract_runtime(archive, destination, "runtime.tar.gz")
    assert (destination / "build/llama-server").read_bytes() == b"test"
    assert os.access(destination / "build/llama-server", os.X_OK)
    assert (destination / "build/libllama.so.1").is_file()
    assert (destination / "build/LICENSE").is_file()
    assert not (destination / "build/llama-bench").exists()
    assert not archive.exists()
    assert (destination / "manifest.json").is_file()


def test_download_is_atomic_and_reuses_verified_file(tmp_path):
    import hashlib

    from sirius_proj.installation import Artifact, download

    source = tmp_path / "source"
    source.write_bytes(b"verified content")
    artifact = Artifact(
        "models/model.bin", source.as_uri(), hashlib.sha256(source.read_bytes()).hexdigest()
    )
    root = tmp_path / "app"
    result = download(artifact, root, lambda text: None, lambda: False)
    source.unlink()
    assert download(artifact, root, lambda text: None, lambda: False) == result
    assert result.read_bytes() == b"verified content"
    assert not result.with_suffix(".bin.part").exists()


def test_failed_checksum_preserves_old_file(tmp_path):
    import pytest

    from sirius_proj.installation import Artifact, download

    source = tmp_path / "source"
    source.write_bytes(b"incomplete model")
    destination = tmp_path / "models/model.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"previous model")
    with pytest.raises(ValueError, match="контрольная сумма"):
        download(
            Artifact("models/model.bin", source.as_uri(), "0" * 64),
            tmp_path,
            lambda text: None,
            lambda: False,
        )
    assert destination.read_bytes() == b"previous model"
    assert not destination.with_suffix(".bin.part").exists()


def test_cancelled_download_cleans_partial_file(tmp_path):
    import hashlib

    import pytest

    from sirius_proj.installation import Artifact, download

    source = tmp_path / "source"
    source.write_bytes(b"a" * (2 * 1024 * 1024))
    artifact = Artifact(
        "models/model.bin", source.as_uri(), hashlib.sha256(source.read_bytes()).hexdigest()
    )
    calls = iter([False, False, True])
    with pytest.raises(InterruptedError):
        download(artifact, tmp_path, lambda text: None, lambda: next(calls))
    assert not (tmp_path / artifact.path).exists()
    assert not (tmp_path / "models/model.bin.part").exists()


def test_frozen_application_uses_writable_user_data(tmp_path, monkeypatch):
    import sys

    from sirius_proj.config import default_data_dir

    monkeypatch.delenv("SIRIUS_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert default_data_dir() == tmp_path / "sirius-review"


def test_split_model_requires_both_shards(tmp_path):
    from sirius_proj.config import default_settings

    settings = default_settings(tmp_path)
    settings.model_file.parent.mkdir(parents=True)
    settings.model_file.touch()
    assert settings.model_file.with_name("model-00002-of-00002.gguf") in settings.missing_components


def test_only_one_installer_can_write_components(tmp_path):
    import pytest

    from sirius_proj.installation import component_lock

    with component_lock(tmp_path):
        with pytest.raises(RuntimeError, match="уже выполняется"):
            with component_lock(tmp_path):
                pytest.fail("Second installer must not write shared files")
    with component_lock(tmp_path):
        pass
