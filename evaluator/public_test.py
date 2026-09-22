"""Post-selection public-test evaluator; never used by the development gate."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import ndcg_score
from core import validate

def qrels(path):
    result = {}
    for row in csv.DictReader(Path(path).open(), delimiter='\t'):
        result.setdefault(row['query-id'], {})[row['corpus-id']] = int(row['score'])
    return result

def manual(ranking, rel, exponential):
    gain = lambda x: (2 ** x - 1) if exponential else x
    dcg = sum(gain(rel.get(doc, 0)) / math.log2(i + 2) for i, doc in enumerate(ranking[:10]))
    ideal = sorted((gain(x) for x in rel.values()), reverse=True)[:10]
    return dcg / sum(x / math.log2(i + 2) for i, x in enumerate(ideal))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--candidate',type=Path,required=True); p.add_argument('--cache',type=Path,required=True); p.add_argument('--qrels',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); a=p.parse_args()
    c=json.loads(a.candidate.read_text()); validate(c); z=np.load(a.cache,allow_pickle=False); docs=[str(x) for x in z['doc_ids']]; labels=qrels(a.qrels); exp=[]; linear=[]; sklearn_exp=[]; sklearn_linear=[]; predictions=[]
    for i,qid0 in enumerate(z['query_ids']):
        qid=str(qid0); scores=c['bm25_weight']/(c['k']+z['bm25_ranks'][i])+(1-c['bm25_weight'])/(c['k']+z['tfidf_ranks'][i]); top=np.argsort(-scores,kind='stable')[:10]; ranking=[docs[j] for j in top]; rel=labels[qid]
        exp.append(manual(ranking,rel,True)); linear.append(manual(ranking,rel,False))
        y=np.array([rel.get(d,0) for d in docs],dtype=float); sklearn_exp.append(ndcg_score((2**y-1)[None,:],scores[None,:],k=10,ignore_ties=True)); sklearn_linear.append(ndcg_score(y[None,:],scores[None,:],k=10,ignore_ties=True))
        predictions.append({'query_id':qid,'top10':ranking})
    a.predictions.write_text(''.join(json.dumps(x)+'\n' for x in predictions))
    report={'split':'public_test','test_labels_read':True,'candidate':c,'query_count':len(predictions),'metrics':{'nDCG_exp@10':float(np.mean(exp)),'nDCG_linear@10':float(np.mean(linear))},'sklearn_per_query_means':{'nDCG_exp@10':float(np.mean(sklearn_exp)),'nDCG_linear@10':float(np.mean(sklearn_linear))},'manual_matches_sklearn':{'exp':bool(np.allclose(exp,sklearn_exp,rtol=0,atol=1e-12)),'linear':bool(np.allclose(linear,sklearn_linear,rtol=0,atol=1e-12))}}
    a.out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
