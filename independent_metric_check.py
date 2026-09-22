"""Cross-check one frozen public-test RRF candidate without importing the evaluator.

This reads locally reconstructed ranks and qrels. It performs a full Python
stable sort for every query and independently computes exponential and linear
nDCG@10. It does not rebuild the ranks or create new agent trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            value = int(row["score"])
            if value < 0:
                raise ValueError("Negative relevance")
            labels.setdefault(row["query-id"], {})[row["corpus-id"]] = value
    return {qid: rel for qid, rel in labels.items() if any(rel.values())}


def gain_numerator(ranking: list[str], rel: dict[str, int], exponential: bool) -> float:
    values = [rel.get(doc, 0) for doc in ranking[:10]]
    if exponential:
        values = [2**v - 1 for v in values]
    return math.fsum(value / math.log2(position + 2) for position, value in enumerate(values))


def ndcg(ranking: list[str], rel: dict[str, int], exponential: bool) -> float:
    ideal = [2**v - 1 if exponential else v for v in rel.values()]
    ideal.sort(reverse=True)
    denom = math.fsum(value / math.log2(position + 2) for position, value in enumerate(ideal[:10]))
    if denom <= 0:
        raise ValueError("Query has no positive relevance")
    return gain_numerator(ranking, rel, exponential) / denom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if set(candidate) != {"k", "bm25_weight"}:
        raise ValueError("Candidate keys changed")
    k, weight = candidate["k"], candidate["bm25_weight"]
    if type(k) is not int or not 1 <= k <= 200 or type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid frozen candidate")

    with np.load(args.cache, allow_pickle=False) as cache:
        qids = [str(item) for item in cache["query_ids"]]
        docs = [str(item) for item in cache["doc_ids"]]
        bm25 = cache["bm25_ranks"]
        tfidf = cache["tfidf_ranks"]
        if docs != sorted(docs) or len(set(docs)) != len(docs):
            raise ValueError("Cache document order is not unique and sorted")
        if len(set(qids)) != len(qids):
            raise ValueError("Duplicate cache query IDs")
        if bm25.shape != tfidf.shape or bm25.shape != (len(qids), len(docs)):
            raise ValueError("Rank arrays have unexpected shape")
        labels = load_qrels(args.qrels)
        if set(labels) != set(qids):
            raise ValueError("Cache and qrels query IDs differ")

        expected_predictions = None
        if args.predictions is not None:
            expected_predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
            if len(expected_predictions) != len(qids):
                raise ValueError("Prediction row count differs")

        exp_scores, linear_scores = [], []
        for row_no, qid in enumerate(qids):
            scores = [
                weight / (k + int(bm25[row_no, doc_no]))
                + (1 - weight) / (k + int(tfidf[row_no, doc_no]))
                for doc_no in range(len(docs))
            ]
            top = sorted(range(len(docs)), key=lambda doc_no: (-scores[doc_no], doc_no))[:10]
            ranking = [docs[doc_no] for doc_no in top]
            if expected_predictions is not None and expected_predictions[row_no] != {"query_id": qid, "top10": ranking}:
                raise ValueError(f"Prediction ranking differs for {qid}")
            exp_scores.append(ndcg(ranking, labels[qid], True))
            linear_scores.append(ndcg(ranking, labels[qid], False))

    result = {
        "status": "PASS_INDEPENDENT_SELECTED_TEST_METRIC",
        "scope": "One frozen public-test candidate; Python stable full sort and independent nDCG, but uses the rebuilt rank cache and public labels. Not blind or cross-machine.",
        "candidate": candidate,
        "query_count": len(qids),
        "document_count": len(docs),
        "nDCG_exp@10": statistics.fmean(exp_scores),
        "nDCG_linear@10": statistics.fmean(linear_scores),
        "prediction_rows_compared": len(qids) if expected_predictions is not None else 0,
        "sha256": {
            "candidate": sha256(args.candidate),
            "cache": sha256(args.cache),
            "qrels": sha256(args.qrels),
            "reference": sha256(args.reference),
            **({"predictions": sha256(args.predictions)} if args.predictions is not None else {}),
        },
    }
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference["candidate"] != candidate or reference["query_count"] != len(qids):
        raise ValueError("Reference metadata differs")
    for key, measured in (("nDCG_exp@10", result["nDCG_exp@10"]), ("nDCG_linear@10", result["nDCG_linear@10"])):
        if not math.isclose(measured, reference["metrics"][key], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Reference metric {key} differs")
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
