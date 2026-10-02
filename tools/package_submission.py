"""Package source and completed artifacts, excluding raw data and private QA."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = ["proposal_ko.pdf", "proposal_en.pdf", "report_ko.pdf", "report_en.pdf",
                "presentation_ko.pptx", "presentation_en.pptx", "execution_summary.json", "test_results.txt",
                "extended/sensitivity/metrics.json", "extended/cnn/metrics.json", "final/model_card.json",
                "predictions/prediction_manifest.json", "extended_verification.json"]
    for name in required:
        if not (ROOT / "output" / name).is_file():
            raise FileNotFoundError(f"Required deliverable is missing: {name}")
    files = [ROOT / name for name in ("README.md", "pyproject.toml", "requirements-tested.txt", "requirements-neural-cpu.txt", "run.ps1", ".gitignore", ".gitattributes")]
    for directory, pattern in (("src", "*.py"), ("tests", "*.py"), ("configs", "*.json"), ("docs", "*.md")):
        files.extend((ROOT / directory).rglob(pattern))
    files.extend(path for path in (ROOT / "tools").iterdir() if path.suffix in {".py", ".mjs", ".ps1"})
    files.extend(path for path in (ROOT / "output").iterdir() if path.suffix in {".pdf", ".docx", ".pptx", ".txt", ".json"})
    for directory in ("results", "benchmark", "provenance", "extended", "final", "predictions", "visuals"):
        files.extend(path for path in (ROOT / "output" / directory).rglob("*") if path.is_file())
    output = ROOT / "output/submission_source.zip"
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(output) as archive:
        failure = archive.testzip()
        if failure is not None:
            raise AssertionError(f"Corrupt archive member: {failure}")
        names = archive.namelist()
        if any(name.startswith("data/") or "__pycache__" in name or ".slides-build" in name for name in names):
            raise AssertionError("Archive contains excluded private or raw files")
    print(f"Packaged {len(files)} files: {output} ({output.stat().st_size:,} bytes); ZIP integrity passed")


if __name__ == "__main__":
    main()
