# Post-release r2 source replay from a retained NFCorpus archive

The fixed [r2 reviewer tag](https://github.com/Zhi-Chao-PAN/research-agent-bench/releases/tag/reviewer-snapshot-2026-09-23-r2) still resolves to `e23760f64c772cf7225280a991fd102740c78def`. After that release, an AI agent cloned the public repository into a new WSL ext4 directory, checked out the tag, created a new Python 3.12.14 environment, and rebuilt the NFCorpus ranking caches. This addendum is **outside the fixed tag**. It uses a retained local copy of the upstream NFCorpus ZIP, with complete SHA-256 `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`; it is **not** evidence of a new network download or a second machine.

## Replayed commands and results

After installing the pinned `requirements-lock.txt`, the agent ran the following commands from the tagged checkout. `raw/nfcorpus.zip` was copied from the earlier local study only after its complete SHA-256 matched `prepare_nfcorpus.py`.

```bash
python prepare_nfcorpus.py --archive raw/nfcorpus.zip --data data
python evaluator/build_cache.py --data data --out cache
python verify_public_traces.py
python verify_phase2.py --results results --out local_phase2_verification.json
python evaluator/public_test.py \
  --candidate fresh-repeat-v5/frozen_unique_candidate.json \
  --cache cache/test_ranks.npz --qrels cache/test_qrels.tsv \
  --out local_selected_test.json --predictions local_selected_top10.jsonl
python synthetic_demo.py
```

The independent checker was **added after r2**, so it is not runnable from the tag alone. The AI agent ran this branch's [`independent_metric_check.py`](independent_metric_check.py) separately against the tagged checkout's rebuilt cache, qrels, archived reference and fresh prediction rows. To repeat that part, run the following from the tagged checkout with `/path/to/addendum/independent_metric_check.py` pointing to this branch's file:

```bash
python /path/to/addendum/independent_metric_check.py \
  --candidate fresh-repeat-v5/frozen_unique_candidate.json \
  --cache cache/test_ranks.npz --qrels cache/test_qrels.tsv \
  --reference fresh-repeat-v5/public_test_selected.json \
  --predictions local_selected_top10.jsonl --out local_independent_check.json
```

The rebuilt cache contains 3,633 documents and 2,590 / 324 / 323 train / development / public-test queries. Its development rank-cache SHA-256 is `7e8430988983b1c538046b520e90a2f5ade719b691d2e766c3640bcaed7c9b88`, matching the earlier [publication-time replay](REPRODUCTION_QA.md). The rebuilt [cache manifest](evidence/post_release_r2_replay_20260923/cache_manifest.json) records complete source-file and cache hashes. The fresh [environment record](evidence/post_release_r2_replay_20260923/environment.json) includes the Python and package versions without a local username or path.

The [full phase-2 verifier output](evidence/post_release_r2_replay_20260923/phase2_verification.json) is `VERIFIED`: it rechecked 189 matrix points, 100 six-call random-search repetitions, 101 development selections, 10,000 paired bootstrap samples, and 348,983 query–candidate instances across 1,078 distinct split–candidate cases. It uses a full stable ranking sort rather than `run_phase2.py`'s partial-selection path, but still imports the shared `ndcg10` metric function. This therefore checks a different ranking path and the archived numbers; it is **not** an independent reimplementation of every scoring component.

For the frozen `k=100, BM25 weight=0.5` selection, [the new public-test aggregate](evidence/post_release_r2_replay_20260923/selected_public_test.json) again reports exponential-gain nDCG@10 `0.3070440739845281` and linear-gain nDCG@10 `0.30640826479410294` on 323 queries. The separate [Python full-sort metric check](independent_metric_check.py) does not import `evaluator/core.py`: it independently calculates gain, discount, ideal ranking, and stable tie order, and compares all 323 top-10 rows with the newly generated predictions and both aggregates with the original archived report. Its [public aggregate output](evidence/post_release_r2_replay_20260923/independent_metric_check.json) is `PASS_INDEPENDENT_SELECTED_TEST_METRIC` (`0.30704407398452815` exponential, a floating-point difference below `1e-15`). This check **shares the rebuilt rank cache** and public labels with the main evaluator; it cannot independently prove the BM25/TF-IDF cache construction was correct.

`verify_public_traces.py` returned `PASS_TRACE_ONLY`; `synthetic_demo.py` returned `PASS_SYNTHETIC_DEMO`, including six fabricated-data calls and a rejected seventh. A new development-only task created from the rebuilt cache accepted one trial with `k=60`, weight `0.5` and returned nDCG@10 `0.2651036767543689`. This was a deterministic tool smoke test by the AI agent, **not** a new six-call LLM trajectory or the applicant's own practice.

## Scope and distribution

The result strengthens the claim that the **public r2 source at its fixed tag** can rebuild the inputs and reproduce the archived numerical analysis in a new software environment on the same machine. It does not replicate the original RRF paper's TREC/LETOR results, make the explored public test blind, authenticate the archive's original network transfer, establish cross-machine reproducibility, or demonstrate the applicant's unaided research skill. The three new agent trajectories still select one identical configuration, scoring `0.307044` below the preset six-point search at `0.307294`; their archived logs have no independently checkable model/provider/session provenance.

The repository does not redistribute NFCorpus text or qrels, derived rank caches, or the 323 top-10 prediction rows. The public files here contain only hashes, counts, aggregate metrics and code. Readers must obtain the upstream archive themselves under the terms described in [DATA_PROVENANCE.md](DATA_PROVENANCE.md); to use the new independent checker, first rebuild `cache/` using the README steps and generate a local predictions file with `evaluator/public_test.py`. The repository's CI still performs offline trace and fabricated-data checks only; it does not run this data-dependent replay.
