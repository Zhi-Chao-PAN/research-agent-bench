"""Fetch the pinned BEIR NFCorpus archive and extract it for local evaluation.

The dataset remains upstream: this repository publishes no source corpus or qrels.
Review the original NFCorpus terms before using it outside academic research.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
SHA256 = "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b"
EXPECTED = {
    "nfcorpus/corpus.jsonl",
    "nfcorpus/queries.jsonl",
    "nfcorpus/qrels/train.tsv",
    "nfcorpus/qrels/dev.tsv",
    "nfcorpus/qrels/test.tsv",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent, prefix="nfcorpus-", suffix=".tmp", delete=False) as temp:
        pending = Path(temp.name)
        try:
            with urlopen(URL, timeout=60) as source:
                shutil.copyfileobj(source, temp)
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    if file_sha256(pending) != SHA256:
        pending.unlink(missing_ok=True)
        raise ValueError("Downloaded archive SHA-256 differs from pinned snapshot")
    pending.replace(archive)


def extract(archive: Path, data: Path) -> None:
    if file_sha256(archive) != SHA256:
        raise ValueError("NFCorpus archive SHA-256 mismatch")
    if data.exists():
        raise FileExistsError(f"Choose a new data directory: {data}")
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        names = {item.filename for item in members}
        if not EXPECTED <= names:
            raise ValueError("Unexpected NFCorpus archive layout")
        for item in members:
            relative = Path(item.filename)
            if relative.is_absolute() or ".." in relative.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive member: {item.filename}")
        with tempfile.TemporaryDirectory(dir=data.parent, prefix="nfcorpus-extract-") as temp:
            staged = Path(temp) / "ready"
            staged.mkdir()
            stream.extractall(staged)
            staged.rename(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("raw/nfcorpus.zip"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="Download the pinned upstream archive if missing")
    args = parser.parse_args()
    if not args.archive.exists():
        if not args.download:
            parser.error("Archive missing; pass --download to fetch the pinned upstream snapshot")
        download(args.archive)
    args.data.parent.mkdir(parents=True, exist_ok=True)
    extract(args.archive, args.data)
    print({"archive_sha256": file_sha256(args.archive), "data": str(args.data)})


if __name__ == "__main__":
    main()
