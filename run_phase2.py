"""CPU<=2 post-hoc matrix and equal-budget random-search analysis."""
import argparse,csv,json,math,os,statistics
from pathlib import Path
os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import numpy as np

def qrels(p):
 d={}
 for r in csv.DictReader(Path(p).open(),delimiter='\t'): d.setdefault(r['query-id'],{})[r['corpus-id']]=int(r['score'])
 return d
def one(cache,labels,c):
 docs=[str(x) for x in cache['doc_ids']]; scores=c['bm25_weight']/(c['k']+cache['bm25_ranks'])+(1-c['bm25_weight'])/(c['k']+cache['tfidf_ranks'])
 # Include every tie at the boundary, then preserve original document-index order.
 partial=np.argpartition(-scores,9,axis=1)[:,:10]
 threshold=np.min(np.take_along_axis(scores,partial,axis=1),axis=1)
 tops=[]
 for row,cutoff in zip(scores,threshold):
  eligible=np.flatnonzero(row>=cutoff)
  tops.append(eligible[np.lexsort((eligible,-row[eligible]))[:10]])
 exp=[]; lin=[]
 for qid,top in zip(cache['query_ids'],tops):
  r=labels[str(qid)]; rank=[docs[i] for i in top]
  def n(expflag):
   gain=lambda x:(2**x-1) if expflag else x; dcg=sum(gain(r.get(x,0))/math.log2(i+2) for i,x in enumerate(rank)); ideal=sorted((gain(x) for x in r.values()),reverse=True)[:10]; return dcg/sum(x/math.log2(i+2) for i,x in enumerate(ideal))
  exp.append(n(True)); lin.append(n(False))
 return {'exp':float(np.mean(exp)),'linear':float(np.mean(lin)),'per_query_exp':exp,'per_query_linear':lin}
def rec(c,x): return {'candidate':c,'dev_exp':x['exp'],'dev_linear':x['linear']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(); pr=json.loads(a.protocol.read_text()); a.out.mkdir(parents=True,exist_ok=True)
 dev=np.load(a.cache/'dev_ranks.npz',allow_pickle=False); test=np.load(a.cache/'test_ranks.npz',allow_pickle=False); dl=qrels(a.cache/'dev_qrels.tsv'); tl=qrels(a.cache/'test_qrels.tsv')
 ks=pr['matrix']['k']; ws=[round(i*.05,2) for i in range(21)]; matrix=[]
 for k in ks:
  for w in ws:
   c={'k':k,'bm25_weight':w}; d=one(dev,dl,c); t=one(test,tl,c); matrix.append({**rec(c,d),'test_exp':t['exp'],'test_linear':t['linear']})
 best=max(enumerate(matrix),key=lambda z:(z[1]['dev_exp'],-z[0]))[1]; dev_order=sorted(matrix,key=lambda x:x['dev_exp'],reverse=True); test_order=sorted(matrix,key=lambda x:x['test_exp'],reverse=True)
 for i,x in enumerate(dev_order,1): x['dev_rank']=i
 test_rank={ (x['candidate']['k'],x['candidate']['bm25_weight']):i for i,x in enumerate(test_order,1)}
 for x in matrix:x['test_rank']=test_rank[(x['candidate']['k'],x['candidate']['bm25_weight'])]
 random=[]
 for seed in range(1000,1100):
  rng=np.random.default_rng(seed); trials=[]
  for _ in range(6):
   c={'k':int(rng.integers(1,201)),'bm25_weight':float(rng.random())}; trials.append(rec(c,one(dev,dl,c)))
  chosen=max(enumerate(trials),key=lambda z:(z[1]['dev_exp'],-z[0]))[1]; t=one(test,tl,chosen['candidate']); random.append({'seed':seed,'trials':trials,'selected':{**chosen,'test_exp':t['exp'],'test_linear':t['linear']}})
 agent=pr['comparators']['historical_agent']; preset=pr['comparators']['preset_six_point_selected']; results={}
 for name,c in {'agent':agent,'preset6':preset}.items():
  d=one(dev,dl,c);t=one(test,tl,c);results[name]={**rec(c,d),'test_exp':t['exp'],'test_linear':t['linear'],'test_per_query_exp':t['per_query_exp'],'test_per_query_linear':t['per_query_linear']}
 rng=np.random.default_rng(pr['bootstrap']['seed']); delta=np.array(results['agent']['test_per_query_exp'])-np.array(results['preset6']['test_per_query_exp']); idx=rng.integers(0,len(delta),(pr['bootstrap']['resamples'],len(delta))); boot=delta[idx].mean(axis=1)
 summary={'protocol':pr,'matrix_count':len(matrix),'matrix_dev_selected':best,'matrix_dev_selected_test_rank':test_rank[(best['candidate']['k'],best['candidate']['bm25_weight'])],'matrix_test_top':test_order[0],'historical':results,'random_search':{'repeats':100,'draws_per_repeat':6,'test_exp_mean':float(np.mean([x['selected']['test_exp'] for x in random])),'test_exp_std':float(np.std([x['selected']['test_exp'] for x in random],ddof=1)),'agent_percentile':float(np.mean([x['selected']['test_exp']<=results['agent']['test_exp'] for x in random]))},'paired_bootstrap_agent_minus_preset6_exp':{'mean':float(delta.mean()),'ci95':[float(x) for x in np.quantile(boot,[.025,.975])]},'selection_uses_development_scores_only':True,'matrix_test_scan_is_post_hoc':True,'random_test_scored_only_for_dev_selected_candidate':True}
 (a.out/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n');(a.out/'random_search_100x6.json').write_text(json.dumps(random,indent=2)+'\n');(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Per-query evidence for primary historical contrast.
 with (a.out/'paired_agent_preset6_per_query.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['query_id','agent_exp','preset6_exp','delta_exp','agent_linear','preset6_linear','delta_linear'])
  for q,a1,b1,a2,b2 in zip(test['query_ids'],results['agent']['test_per_query_exp'],results['preset6']['test_per_query_exp'],results['agent']['test_per_query_linear'],results['preset6']['test_per_query_linear']):w.writerow([q,a1,b1,a1-b1,a2,b2,a2-b2])
if __name__=='__main__':main()
