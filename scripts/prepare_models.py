"""Explicit online preparation. The application itself never downloads models."""

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = "b12420"


def get_json(url):
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "sirius-2026"}), timeout=60
    ) as stream:
        return json.load(stream)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, expected=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (expected is None or digest(destination) == expected):
        print(f"Уже подготовлен: {destination.name}", flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "sirius-2026"})
    print(f"Загрузка: {destination.name}", flush=True)
    with urllib.request.urlopen(request, timeout=180) as source, temporary.open("wb") as target:
        while block := source.read(1024 * 1024):
            target.write(block)
    if expected and digest(temporary) != expected:
        temporary.unlink()
        raise ValueError(f"Не совпала SHA256: {destination.name}")
    os.replace(temporary, destination)
    print(f"Готово: {destination.name}", flush=True)


def model_tasks(repo, files, folder):
    metadata = get_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
    revision = metadata["sha"]
    index = {item["rfilename"]: item for item in metadata["siblings"]}
    tasks, manifest = [], {"repository": repo, "revision": revision, "files": {}}
    for remote, local in files.items():
        expected = index[remote].get("lfs", {}).get("sha256")
        tasks.append(
            (f"https://huggingface.co/{repo}/resolve/{revision}/{remote}", folder / local, expected)
        )
        manifest["files"][local] = expected
    return tasks, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Подготовить CPU-сборку вместо Vulkan")
    parser.add_argument("--model-size", choices=["3b", "7b"], default="7b")
    args = parser.parse_args()
    e5 = ROOT / "models" / "e5"
    llm = ROOT / "models" / ("qwen7b" if args.model_size == "7b" else "qwen")
    tasks, e5_manifest = model_tasks(
        "intfloat/multilingual-e5-small",
        {
            "onnx/model.onnx": "model.onnx",
            "tokenizer.json": "tokenizer.json",
        },
        e5,
    )
    files = {"qwen2.5-3b-instruct-q4_k_m.gguf": "model.gguf"}
    repo = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    if args.model_size == "7b":
        repo = "Qwen/Qwen2.5-7B-Instruct-GGUF"
        files = {
            f"qwen2.5-7b-instruct-q4_k_m-0000{i}-of-00002.gguf": f"model-0000{i}-of-00002.gguf"
            for i in (1, 2)
        }
    llm_tasks, llm_manifest = model_tasks(repo, files, llm)
    variant = "ubuntu-x64" if args.cpu else "ubuntu-vulkan-x64"
    name = f"llama-{BUILD}-bin-{variant}.tar.gz"
    release = get_json(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{BUILD}")
    asset = next(item for item in release["assets"] if item["name"] == name)
    archive = ROOT / "runtime" / name
    expected = asset.get("digest", "") or ""
    expected = expected.removeprefix("sha256:") or None
    tasks += llm_tasks + [(asset["browser_download_url"], archive, expected)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, *task) for task in tasks]
        for future in futures:
            future.result()
    runtime = ROOT / "runtime" / "llama"
    runtime.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as package:
        package.extractall(runtime, filter="data")
    for folder, manifest in [(e5, e5_manifest), (llm, llm_manifest)]:
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Модели подготовлены. Для запуска: uv run sirius-proj", flush=True)


if __name__ == "__main__":
    main()
