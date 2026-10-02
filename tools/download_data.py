"""Download the public DS02 redistribution with resuming and checksum validation.

Original dataset: NASA N-CMAPSS. Delivery mirror: Figshare 20436504 v1.
Figshare's published MD5 validates this mirror file, not equivalence to NASA's ZIP.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = "https://api.figshare.com/v2/articles/20436504/versions/1"
EXPECTED_ID = 36563133
EXPECTED_MD5 = "61056251b36290e11371e017eed70eac"
EXPECTED_SIZE = 2450472504
NAME = "N-CMAPSS_DS02-006.h5"
NASA = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"


def checksum(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output: Path, workers: int = 6) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    meta = requests.get(ARTICLE, timeout=30)
    meta.raise_for_status()
    article = meta.json()
    file = next(f for f in article["files"] if f["id"] == EXPECTED_ID)
    if file["size"] != EXPECTED_SIZE or file["computed_md5"] != EXPECTED_MD5:
        raise RuntimeError("Mirror metadata changed; inspect provenance before downloading.")
    target = output / NAME
    if target.exists():
        if target.stat().st_size != EXPECTED_SIZE or checksum(target) != EXPECTED_MD5:
            raise RuntimeError(f"Existing file fails checksum: {target}; preserving it.")
        print("Existing DS02 checksum verified", flush=True)
    else:
        partdir = output / (NAME + ".parts")
        partdir.mkdir(exist_ok=True)
        partsize = 64 * 1024 * 1024
        pieces = [(i, begin, min(begin + partsize, EXPECTED_SIZE) - 1)
                  for i, begin in enumerate(range(0, EXPECTED_SIZE, partsize))]

        def fetch(piece: tuple[int, int, int]) -> tuple[int, int]:
            index, begin, end = piece
            destination = partdir / f"{index:04d}.part"
            needed = end - begin + 1
            if destination.exists() and destination.stat().st_size == needed:
                return index, needed
            for attempt in range(4):
                try:
                    temporary = destination.with_suffix(".partial")
                    with requests.get(file["download_url"], headers={"Range": f"bytes={begin}-{end}"},
                                      timeout=(30, 120), stream=True) as response:
                        response.raise_for_status()
                        actual = response.headers.get("Content-Range", "")
                        if response.status_code != 206 or actual != f"bytes {begin}-{end}/{EXPECTED_SIZE}":
                            raise RuntimeError(f"Unexpected range response: {response.status_code} {actual}")
                        with temporary.open("wb") as handle:
                            for block in response.iter_content(1024 * 1024):
                                if block:
                                    handle.write(block)
                    if temporary.stat().st_size != needed:
                        raise RuntimeError("Incomplete range response")
                    temporary.replace(destination)
                    return index, needed
                except (requests.RequestException, RuntimeError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            raise AssertionError("Unreachable")

        started = time.perf_counter()
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, piece) for piece in pieces]
            for future in as_completed(futures):
                _, size = future.result()
                completed += size
                elapsed = max(time.perf_counter() - started, .001)
                print(f"{completed / EXPECTED_SIZE:.1%} {completed / 1e9:.2f} GB "
                      f"{completed / elapsed / 1e6:.1f} MB/s", flush=True)
        temporary = target.with_suffix(".h5.partial")
        digest = hashlib.md5()
        with temporary.open("wb") as result:
            for index, _, _ in pieces:
                with (partdir / f"{index:04d}.part").open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                        result.write(block)
        if digest.hexdigest() != EXPECTED_MD5:
            raise RuntimeError("Downloaded file failed the published MD5 check; partial files preserved.")
        temporary.replace(target)
        # Remove only this downloader's verified part files, retaining the final data.
        for index, _, _ in pieces:
            (partdir / f"{index:04d}.part").unlink()
        if not any(partdir.iterdir()):
            partdir.rmdir()
    manifest = {
        "dataset": NAME, "source_type": "third_party_redistribution",
        "original_source": NASA, "original_paper": "https://doi.org/10.3390/data6010005",
        "mirror_doi": "https://doi.org/10.6084/m9.figshare.20436504.v1",
        "mirror_author": "Hao Li", "download_url": file["download_url"],
        "bytes": target.stat().st_size, "md5": EXPECTED_MD5,
        "sha256": checksum(target, "sha256"),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_scope": "Matches Figshare published MD5; not independently compared byte-for-byte with NASA archive.",
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified: {target}", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")
    download(args.output, args.workers)
