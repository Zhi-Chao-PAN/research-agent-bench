"""Create a fresh, development-only six-call task from local rebuilt caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "fresh-repeat-v5/fresh-dev-agent-01"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--out", type=Path, default=ROOT / "new-agent-dev")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"Choose a new trajectory directory: {args.out}")
    if args.out.parent.resolve() != ROOT:
        raise ValueError("The task directory must be a direct child of this repository")
    source = (TEMPLATE / "TASK.md").read_text()
    before, marker, tail = source.partition("```json\n")
    if not marker:
        raise ValueError("Template protocol block missing")
    payload, marker, after = tail.partition("\n```")
    if not marker:
        raise ValueError("Template protocol block incomplete")
    protocol = json.loads(payload)
    for filename in ("dev_ranks.npz", "dev_qrels.tsv"):
        if not (args.cache / filename).is_file():
            raise FileNotFoundError(args.cache / filename)
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".new-agent-task-") as temporary:
        staged = Path(temporary) / "task"
        staged.mkdir()
        for directory in ("data", "candidates", "runs"):
            (staged / directory).mkdir()
        for filename in ("dev_ranks.npz", "dev_qrels.tsv"):
            shutil.copy2(args.cache / filename, staged / "data" / filename)
        shutil.copy2(TEMPLATE / "supervisor.py", staged / "supervisor.py")
        protocol["immutable_sha256"] = {
            "data/dev_ranks.npz": sha(staged / "data/dev_ranks.npz"),
            "data/dev_qrels.tsv": sha(staged / "data/dev_qrels.tsv"),
            "../evaluator/core.py": sha(ROOT / "evaluator/core.py"),
            "../evaluator/evaluate_candidate.py": sha(ROOT / "evaluator/evaluate_candidate.py"),
        }
        task = before + "```json\n" + json.dumps(protocol, ensure_ascii=False, separators=(",", ":")) + "\n```" + after
        (staged / "TASK.md").write_text(task)
        staged.rename(args.out)
    print(json.dumps({"task": str(args.out), "immutable_sha256": protocol["immutable_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
