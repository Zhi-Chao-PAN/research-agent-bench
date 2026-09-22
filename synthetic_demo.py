"""Exercise the real six-call supervisor on a tiny, fabricated rank cache.

The fixture is deliberately unrelated to NFCorpus and is never research evidence.
It lets reviewers see development feedback, the budget gate, and selection
without downloading any third-party data.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parent


def write_fixture(cache: Path) -> None:
    """Three queries, twelve documents; two queries favor BM25, one TF-IDF."""
    docs = [f"d{number:02d}" for number in range(1, 13)]
    qids = ["q1", "q2", "q3"]
    bm25 = np.empty((3, 12), dtype=np.uint16)
    tfidf = np.empty_like(bm25)
    for row, relevant_doc in enumerate((0, 1, 2)):
        others = [index for index in range(12) if index != relevant_doc]
        for rank, index in enumerate([relevant_doc, *others], start=1):
            bm25[row, index] = rank
        for rank, index in enumerate([*others, relevant_doc], start=1):
            tfidf[row, index] = rank
    # The third query reverses which ranker puts the answer first.
    bm25[2], tfidf[2] = tfidf[2].copy(), bm25[2].copy()
    np.savez_compressed(
        cache / "dev_ranks.npz",
        query_ids=np.array(qids),
        doc_ids=np.array(docs),
        bm25_ranks=bm25,
        tfidf_ranks=tfidf,
    )
    (cache / "dev_qrels.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td01\t1\nq2\td02\t1\nq3\td03\t1\n",
        encoding="utf-8",
    )


def run_trial(task: Path, number: int, candidate: dict, hypothesis: str) -> dict:
    candidate_path = task / "candidates" / f"{number:02d}.json"
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(task / "supervisor.py"), "--candidate", str(candidate_path),
         "--hypothesis", hypothesis],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep the generated task for manual inspection")
    args = parser.parse_args()
    task = ROOT / f".synthetic-task-{uuid4().hex}"
    try:
        with tempfile.TemporaryDirectory(prefix="synthetic-cache-") as temporary:
            cache = Path(temporary)
            write_fixture(cache)
            subprocess.run(
                [sys.executable, str(ROOT / "create_dev_task.py"),
                 "--cache", str(cache), "--out", str(task)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
        task_doc = task / "TASK.md"
        task_doc.write_text(
            task_doc.read_text(encoding="utf-8").replace(
                "BEIR NFCorpus", "fabricated synthetic fixture (not NFCorpus)"
            ).replace("NFCorpus", "synthetic fixture"),
            encoding="utf-8",
        )
        proposals = [
            ({"k": 60, "bm25_weight": 0.0}, "Check the TF-IDF-only ranking on three fabricated queries."),
            ({"k": 60, "bm25_weight": 1.0}, "Check whether BM25 ranks the relevant document earlier."),
            ({"k": 10, "bm25_weight": 0.5}, "Compare an equal-weight fusion with a smaller rank offset."),
            ({"k": 100, "bm25_weight": 0.5}, "Compare an equal-weight fusion with a larger rank offset."),
            ({"k": 60, "bm25_weight": 0.25}, "Shift a quarter of the weight toward the BM25 ranker."),
            ({"k": 60, "bm25_weight": 0.75}, "Shift three quarters of the weight toward BM25."),
        ]
        trials = [run_trial(task, index, proposal, hypothesis)
                  for index, (proposal, hypothesis) in enumerate(proposals, start=1)]
        if any(trial["status"] != "SUCCESS" for trial in trials):
            raise AssertionError("synthetic trial failed")
        a = trials[0]["metrics"]["ndcg@10"]
        b = trials[1]["metrics"]["ndcg@10"]
        if not math.isclose(a, 1 / 3, abs_tol=1e-12) or not math.isclose(b, 2 / 3, abs_tol=1e-12):
            raise AssertionError(f"unexpected synthetic nDCG scores: {a}, {b}")
        state = json.loads((task / "state.json").read_text(encoding="utf-8"))
        if len(state["trials"]) != 6:
            raise AssertionError("trial state did not preserve the six-call budget")
        winner = max(state["trials"], key=lambda trial: (trial["metrics"]["ndcg@10"], -trial["trial"]))
        rejected = subprocess.run(
            [sys.executable, str(task / "supervisor.py"),
             "--candidate", str(task / "candidates" / "01.json"),
             "--hypothesis", "Try a seventh call to verify the budget gate."],
            cwd=ROOT, text=True, capture_output=True,
        )
        if rejected.returncode == 0 or "six-trial or 15-minute budget exhausted" not in rejected.stderr:
            raise AssertionError("seventh call was not rejected by the budget gate")
        print(json.dumps({
            "status": "PASS_SYNTHETIC_DEMO",
            "dataset": "fabricated 3-query, 12-document fixture; no NFCorpus data or test scores",
            "calls_used": len(state["trials"]),
            "development_ndcg_at_10": {"tfidf_only": a, "bm25_only": b},
            "chosen_from_observed_development_feedback": winner["candidate"],
            "seventh_call_rejected": True,
            "task_directory": str(task) if args.keep else "removed after verification",
        }, indent=2))
    finally:
        if not args.keep and task.exists():
            shutil.rmtree(task)


if __name__ == "__main__":
    main()
