"""Independently recompute every matrix/search score using full stable sorting."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'evaluator'))
from core import read_qrels, ndcg10


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=ROOT / 'results')
    p.add_argument('--out', type=Path, default=ROOT / 'final_verification.json')
    args = p.parse_args()
    protocol = json.loads((ROOT / 'phase2_protocol.json').read_text())
    cached = {split: np.load(ROOT / f'cache/{split}_ranks.npz', allow_pickle=False) for split in ('dev', 'test')}
    labels = {split: read_qrels(ROOT / f'cache/{split}_qrels.tsv') for split in cached}
    rows_checked = cases = 0
    memo = {}
    def independent(split, candidate):
        nonlocal rows_checked, cases
        key = (split, candidate['k'], candidate['bm25_weight'])
        if key in memo:
            return memo[key]
        data = cached[split]
        docs = data['doc_ids'].astype(str)
        require(list(docs) == sorted(docs), 'Document tie-order changed')
        weight, k = candidate['bm25_weight'], candidate['k']
        score = weight / (k + data['bm25_ranks']) + (1 - weight) / (k + data['tfidf_ranks'])
        rankings = np.argsort(-score, axis=1, kind='stable')[:, :10]
        exponential, linear = [], []
        discounts = np.log2(np.arange(2, 12))
        for qid, top in zip(data['query_ids'], rankings):
            rel = labels[split][str(qid)]
            ranked = [str(docs[index]) for index in top]
            exponential.append(ndcg10(ranked, rel))
            gains = np.array([rel.get(doc, 0) for doc in ranked], dtype=float)
            ideal = np.array(sorted(rel.values(), reverse=True)[:10], dtype=float)
            linear.append(float((gains / discounts).sum() / (ideal / discounts[:len(ideal)]).sum()))
        result = {'exp': float(np.mean(exponential)), 'linear': float(np.mean(linear)), 'per_query_exp': np.array(exponential)}
        rows_checked += len(rankings)
        cases += 1
        memo[key] = result
        return result
    def check_record(row, split):
        recalculated = independent(split, row['candidate'])
        for metric in ('exp', 'linear'):
            require(abs(recalculated[metric] - row[f'{split}_{metric}']) < 1e-12, f'{split} metric mismatch: {row["candidate"]}')
    matrix = json.loads((args.results / 'matrix.json').read_text())
    random = json.loads((args.results / 'random_search_100x6.json').read_text())
    summary = json.loads((args.results / 'summary.json').read_text())
    expected = [(k, round(i * .05, 2)) for k in protocol['matrix']['k'] for i in range(21)]
    require([(row['candidate']['k'], row['candidate']['bm25_weight']) for row in matrix] == expected, 'Grid configurations/order mismatch')
    for row in matrix:
        check_record(row, 'dev')
        check_record(row, 'test')
    chosen = max(enumerate(matrix), key=lambda pair: (pair[1]['dev_exp'], -pair[0]))[1]
    require(chosen['candidate'] == summary['matrix_dev_selected']['candidate'], 'Grid winner selected using wrong split')
    require([row['seed'] for row in random] == list(range(1000, 1100)), 'Random seeds missing')
    for trajectory in random:
        generator = np.random.default_rng(trajectory['seed'])
        require(len(trajectory['trials']) == 6, 'Unequal random budget')
        for trial in trajectory['trials']:
            candidate = {'k': int(generator.integers(1, 201)), 'bm25_weight': float(generator.random())}
            require(candidate == trial['candidate'], 'Random draw differs')
            check_record(trial, 'dev')
        winner = max(enumerate(trajectory['trials']), key=lambda pair: (pair[1]['dev_exp'], -pair[0]))[1]
        require(trajectory['selected']['candidate'] == winner['candidate'], 'Random winner differs')
        check_record(trajectory['selected'], 'test')
    historical = summary['historical']
    for result in historical.values():
        check_record(result, 'dev')
        check_record(result, 'test')
        require(np.allclose(result['test_per_query_exp'], independent('test', result['candidate'])['per_query_exp'], atol=1e-14, rtol=0), 'Historical query vector differs')
    delta = independent('test', historical['agent']['candidate'])['per_query_exp'] - independent('test', historical['preset6']['candidate'])['per_query_exp']
    generator = np.random.default_rng(protocol['bootstrap']['seed'])
    samples = generator.integers(0, len(delta), (protocol['bootstrap']['resamples'], len(delta)))
    interval = np.quantile(delta[samples].mean(axis=1), [.025, .975])
    bootstrap = summary['paired_bootstrap_agent_minus_preset6_exp']
    require(np.allclose(interval, bootstrap['ci95'], rtol=0, atol=1e-14), 'Bootstrap interval differs')
    require(abs(delta.mean() - bootstrap['mean']) < 1e-14, 'Bootstrap mean differs')
    scores = np.array([entry['selected']['test_exp'] for entry in random])
    random_summary = summary['random_search']
    require(abs(scores.mean() - random_summary['test_exp_mean']) < 1e-14 and abs(scores.std(ddof=1) - random_summary['test_exp_std']) < 1e-14, 'Random aggregate differs')
    require(summary['selection_uses_development_scores_only'] and summary['matrix_test_scan_is_post_hoc'], 'Misleading test metadata')
    require(abs(np.mean(scores <= historical['agent']['test_exp']) - random_summary['agent_percentile']) < 1e-14, 'Agent percentile differs')
    report = {'status': 'VERIFIED', 'matrix_points': len(matrix), 'random_search_repeats': len(random), 'random_development_calls': 600, 'distinct_split_candidate_cases_recomputed': cases, 'query_candidate_cases_recomputed': rows_checked, 'reference_ranking': 'Full np.argsort stable, independent of partial-selection implementation', 'both_gain_definitions_recomputed': True, 'development_selections_recomputed': 101, 'bootstrap_resamples_recomputed': protocol['bootstrap']['resamples'], 'cache_sha256': {path.name: sha(path) for path in sorted((ROOT / 'cache').glob('*')) if path.is_file()}, 'source_sha256': {name: sha(ROOT / name) for name in ('run_phase2.py', 'verify_phase2.py', 'phase2_protocol.json', 'evaluator/core.py')}, 'scope': 'Post-hoc search and metric analysis; additional LLM trajectories NOT_RUN'}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
