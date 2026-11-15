"""Export defense documents using an isolated LibreOffice profile."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "defense"


def main():
    if not shutil.which("libreoffice"):
        raise SystemExit(
            "Для экспорта нужен LibreOffice. DOCX/PPTX уже доступны для редактирования."
        )
    with tempfile.TemporaryDirectory(prefix="sirius-office-") as profile:
        command = ["libreoffice", "-env:UserInstallation=" + Path(profile).as_uri(), "--headless"]
        files = sorted(DEST.glob("*.docx")) + sorted(DEST.glob("*.pptx"))
        subprocess.run(
            command
            + ["--convert-to", "pdf", "--outdir", str(DEST)]
            + [str(path) for path in files],
            check=True,
            timeout=120,
        )
        subprocess.run(
            command
            + [
                "--convert-to",
                "doc:MS Word 97",
                "--outdir",
                str(DEST),
                str(DEST / "Аннотация.docx"),
            ],
            check=True,
            timeout=60,
        )
    for path in sorted(DEST.glob("*.pdf")):
        pages = len(PdfReader(path).pages)
        if path.stem == "Аннотация" and pages != 1:
            raise SystemExit("Аннотация должна занимать одну страницу")
        if path.stem == "Презентация" and pages > 15:
            raise SystemExit("Презентация должна содержать не более 15 слайдов")
        print(f"{path.name}: {pages} стр.")


if __name__ == "__main__":
    main()
