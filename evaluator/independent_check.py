"""Independent graded nDCG@10 check; intentionally does not import core.metrics."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--qrels',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(); c=json.loads(a.candidate.read_text()); z=np.load(a.cache,allow_pickle=False); rel={}
for r in csv.DictReader(a.qrels.open(),delimiter='\t'): rel.setdefault(r['query-id'],{})[r['corpus-id']]=int(r['score'])
docs=list(z['doc_ids']); vals=[]
for i,qid in enumerate(z['query_ids']):
 scores=c['bm25_weight']/(c['k']+z['bm25_ranks'][i])+(1-c['bm25_weight'])/(c['k']+z['tfidf_ranks'][i]); rank=[docs[j] for j in np.argsort(-scores,kind='stable')[:10]]; r=rel[str(qid)]
 dcg=sum(((2**r.get(doc,0)-1)/math.log2(j+2)) for j,doc in enumerate(rank)); ideal=sorted((2**v-1 for v in r.values()),reverse=True)[:10]; vals.append(dcg/sum(v/math.log2(j+2) for j,v in enumerate(ideal)))
a.out.write_text(json.dumps({'independent_ndcg@10':float(np.mean(vals))},indent=2)+'\n')
