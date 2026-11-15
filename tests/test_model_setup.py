import io
import os
import runpy
import tarfile


def test_runtime_install_keeps_server_libraries_and_license_without_archive(tmp_path):
    install = runpy.run_path("scripts/prepare_models.py")["extract_runtime"]
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        for name in ["llama-server", "libllama.so.1", "LICENSE", "llama-bench"]:
            item = tarfile.TarInfo("build/" + name)
            item.size = 4
            item.mode = 0o755
            package.addfile(item, io.BytesIO(b"test"))
    destination = tmp_path / "runtime"
    install(archive, destination, "runtime.tar.gz")
    assert (destination / "build/llama-server").read_bytes() == b"test"
    assert os.access(destination / "build/llama-server", os.X_OK)
    assert (destination / "build/libllama.so.1").is_file()
    assert (destination / "build/LICENSE").is_file()
    assert not (destination / "build/llama-bench").exists()
    assert not archive.exists()
    assert (destination / "manifest.json").is_file()
