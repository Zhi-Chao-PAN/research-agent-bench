"""Frozen CPU-only weighted-RRF evaluation primitives for NFCorpus."""
import csv, hashlib, json, math, os
from pathlib import Path
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
import numpy as np

THREAD_LIMIT = 2

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def read_qrels(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            score = int(r["score"])
            if score < 0:
                raise ValueError("negative relevance")
            rows.setdefault(r["query-id"], {})[r["corpus-id"]] = score
    return {q: rel for q, rel in rows.items() if any(rel.values())}

def ndcg10(ranking, relevant, cutoff=10):
    if len(ranking) != len(set(ranking)) or not relevant:
        raise ValueError("invalid ranking or empty labels")
    gains = [((2 ** relevant.get(doc, 0)) - 1) / math.log2(i + 2) for i, doc in enumerate(ranking[:cutoff])]
    ideal = sorted(((2 ** x) - 1 for x in relevant.values()), reverse=True)[:cutoff]
    denom = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return sum(gains) / denom if denom else 0.0

def metrics(ranking, relevant):
    positives = {d for d, s in relevant.items() if s > 0}
    hits = [d in positives for d in ranking[:10]]
    return {"ndcg@10": ndcg10(ranking, relevant), "recall@10": sum(hits) / len(positives),
            "mrr@10": next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)}

def validate(candidate):
    if set(candidate) != {"k", "bm25_weight"}:
        raise ValueError("candidate must contain only k and bm25_weight")
    if type(candidate["k"]) is not int or not 1 <= candidate["k"] <= 200:
        raise ValueError("k must be integer 1..200")
    w = candidate["bm25_weight"]
    if type(w) not in (int, float) or not math.isfinite(w) or not 0 <= w <= 1:
        raise ValueError("bm25_weight must be finite in [0,1]")

def evaluate_loaded(cache, qrels, candidate):
    validate(candidate)
    qids = [str(x) for x in cache["query_ids"]]
    docs = [str(x) for x in cache["doc_ids"]]
    if set(qids) != set(qrels):
        raise ValueError("cache and qrels query IDs differ")
    k, w = candidate["k"], candidate["bm25_weight"]
    scores = w / (k + cache["bm25_ranks"]) + (1 - w) / (k + cache["tfidf_ranks"])
    top_rows = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    values = [metrics([docs[j] for j in top], qrels[qid]) for qid, top in zip(qids, top_rows)]
    return {name: float(np.mean([x[name] for x in values])) for name in values[0]}

def evaluate(cache_path, qrels_path, candidate):
    return evaluate_loaded(np.load(cache_path, allow_pickle=False), read_qrels(qrels_path), candidate)
