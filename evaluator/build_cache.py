"""Build immutable BM25/TF-IDF rank caches before labels are evaluated."""
import argparse, json, os, re
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from core import read_jsonl, read_qrels, sha256

def tokens(text): return re.findall(r"[a-z0-9]+", text.lower())

def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--out", type=Path, required=True); args = p.parse_args()
    root = args.data / "nfcorpus"; corpus = sorted(read_jsonl(root / "corpus.jsonl"), key=lambda x: x["_id"])
    docs = [x["_id"] for x in corpus]; text = [(x.get("title", "") + " " + x["text"]).strip() for x in corpus]
    queries = {x["_id"]: x["text"] for x in read_jsonl(root / "queries.jsonl")}
    bm25 = BM25Okapi([tokens(x) for x in text]); vec = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vec.fit_transform(text); args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"thread_limit": 2, "corpus": len(docs), "source_sha256": {str(x.relative_to(args.data)): sha256(x) for x in sorted(root.rglob("*")) if x.is_file()}}
    for split in ("train", "dev", "test"):
        qrels = read_qrels(root / "qrels" / f"{split}.tsv"); qids = sorted(qrels)
        a = np.empty((len(qids), len(docs)), dtype=np.uint16); b = np.empty_like(a)
        for i, qid in enumerate(qids):
            a[i, np.argsort(-bm25.get_scores(tokens(queries[qid])), kind="stable")] = np.arange(1, len(docs)+1)
            scores = (matrix @ vec.transform([queries[qid]]).T).toarray().ravel()
            b[i, np.argsort(-scores, kind="stable")] = np.arange(1, len(docs)+1)
        np.savez_compressed(args.out / f"{split}_ranks.npz", query_ids=np.array(qids), doc_ids=np.array(docs), bm25_ranks=a, tfidf_ranks=b)
        (args.out / f"{split}_qrels.tsv").write_bytes((root / "qrels" / f"{split}.tsv").read_bytes())
        manifest[split] = {"queries": len(qids), "cache_sha256": sha256(args.out / f"{split}_ranks.npz"), "qrels_sha256": sha256(args.out / f"{split}_qrels.tsv")}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__": main()
