# Fresh BEIR archive replay of the fixed r2 source

This is a **post-release, same-machine AI-agent replay**, separate from the fixed [r2 reviewer snapshot](https://github.com/Zhi-Chao-PAN/research-agent-bench/releases/tag/reviewer-snapshot-2026-09-23-r2) at `e23760f64c772cf7225280a991fd102740c78def`. The earlier [source replay](POST_RELEASE_R2_SOURCE_REPLAY_2026-09-23.md) started from a retained local archive. On 2026-09-23, an AI agent made a new clean checkout at that r2 tag, created a Python 3.12.14 environment, and invoked `prepare_nfcorpus.py --download` with an absent local archive. The command fetched the pinned **BEIR-converted NFCorpus ZIP** from the URL in [data provenance](DATA_PROVENANCE.md); this 2,448,432-byte snapshot is not the original Heidelberg full-corpus distribution. The locally observed ZIP SHA-256 was `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`, which the download script enforces before extraction.

The new run removes one limitation of the earlier *retained-archive* replay: the agent did not copy the input ZIP from that prior local study. It is still a local execution observation, **not independently authenticated network provenance**, a second-machine run, a blind test, or work performed personally by the applicant. The same archive hash also means both runs used the same pinned data snapshot, not independent samples.

## Commands and observed scope

From the clean r2 checkout, after installing `requirements-lock.txt` in a fresh environment, the agent ran:

```bash
.venv/bin/python prepare_nfcorpus.py --download --archive raw/nfcorpus.zip --data data
.venv/bin/python evaluator/build_cache.py --data data --out cache
.venv/bin/python verify_phase2.py --results results --out reproduced/phase2_verification.json
.venv/bin/python evaluator/public_test.py \
  --candidate fresh-repeat-v5/frozen_unique_candidate.json \
  --cache cache/test_ranks.npz --qrels cache/test_qrels.tsv \
  --out reproduced/selected_public_test.json \
  --predictions reproduced/selected_top10.jsonl
.venv/bin/python verify_public_traces.py
```

The post-r2 [`independent_metric_check.py`](independent_metric_check.py) was then run separately against the rebuilt cache, qrels, archived reference and 323 newly generated prediction rows. This checker is **not** contained in the r2 tag; copy it from current `main` to a separate path before running the following from the tagged checkout:

```bash
.venv/bin/python /path/to/current-main/independent_metric_check.py \
  --candidate fresh-repeat-v5/frozen_unique_candidate.json \
  --cache cache/test_ranks.npz --qrels cache/test_qrels.tsv \
  --reference fresh-repeat-v5/public_test_selected.json \
  --predictions reproduced/selected_top10.jsonl \
  --out reproduced/independent_selected_metric.json
```

The environment used Python 3.12.14, NumPy 2.2.6, scikit-learn 1.6.1 and rank-bm25 0.2.2.

| Rechecked item | Fresh result | Comparison with earlier retained-archive replay |
|---|---|---|
| BEIR NFCorpus transformed ZIP | 2,448,432 bytes; pinned SHA-256 matched | Same pinned archive bytes; locally fetched anew |
| Rebuilt rank cache | 3,633 documents; train/dev/test queries 2,590/324/323 | [Manifest](evidence/public_beir_replay_20260923/cache_manifest.json) byte-identical to earlier replay |
| Phase-2 analysis | `VERIFIED`: 189 matrix points, 100 six-call random searches, 10,000 paired bootstrap samples, 348,983 query–candidate cases | [Verifier JSON](evidence/public_beir_replay_20260923/phase2_verification.json) byte-identical |
| Frozen selection public test | `k=100`, BM25 weight `0.5`; exponential-gain nDCG@10 `0.3070440739845281` over 323 queries | [Evaluator JSON](evidence/public_beir_replay_20260923/selected_public_test.json) byte-identical |
| Separate selected-candidate metric check | `PASS_INDEPENDENT_SELECTED_TEST_METRIC`; exponential nDCG@10 `0.30704407398452815`; 323 top-10 rows matched | [Checker JSON](evidence/public_beir_replay_20260923/independent_selected_metric.json) byte-identical |

The independent checker reimplements gain, discount, ideal ranking and stable tie order without importing the evaluator metric function. It **still shares the rebuilt rank cache and public labels**, so it does not independently validate rank-cache construction or make the already explored public test blind. `verify_public_traces.py` returned `PASS_TRACE_ONLY`; that script checks trace records without rerunning LLM calls. The three additional six-call trajectories still collapse to one selected ranking and its `0.307044` result remains below the six-point preset search at `0.307294`. There is no new agent trajectory and no changed research conclusion.

The public [asset observation](evidence/public_beir_replay_20260923/asset_observation.json) records the command, URL, size, digest and versions; it is self-reported local execution evidence. [SHA256SUMS](evidence/public_beir_replay_20260923/SHA256SUMS) covers the five small records. This repository does **not** redistribute the third-party archive, source text, qrels, rank caches or 323 prediction rows. Readers who reproduce the run must obtain the archive upstream and check its applicable terms. CI checks trace records, static hashes and a synthetic task offline; it does not perform this data-dependent replay.
