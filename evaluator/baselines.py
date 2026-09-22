"""Fixed six-point grid and five reproducible six-draw random searches."""
import argparse,json
from pathlib import Path
import numpy as np
from core import evaluate_loaded, read_qrels
p=argparse.ArgumentParser(); p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--kind',choices=('grid','random'),required=True);p.add_argument('--seed',type=int);a=p.parse_args(); d=a.cache_dir
bundles={s:(np.load(d/f'{s}_ranks.npz',allow_pickle=False),read_qrels(d/f'{s}_qrels.tsv')) for s in ('dev','test')}
def ev(c,split): return evaluate_loaded(*bundles[split],c)
grid=[{"k":1,"bm25_weight":0.0},{"k":10,"bm25_weight":0.2},{"k":30,"bm25_weight":0.4},{"k":60,"bm25_weight":0.6},{"k":120,"bm25_weight":0.8},{"k":200,"bm25_weight":1.0}]
def choose(xs): return max(enumerate(xs),key=lambda x:(x[1]['dev']['ndcg@10'],-x[0]))[1]
if a.kind=='grid':
 rows=[{"candidate":c,"dev":ev(c,'dev')} for c in grid]; result={"grid_6":rows,"selected":choose(rows)}; result["selected"]["public_test_after_selection"]=ev(result["selected"]["candidate"],'test')
else:
 if a.seed is None: raise SystemExit('--seed required for random')
 rng=np.random.default_rng(a.seed); seen=set(); rows=[]
 while len(rows)<6:
  c={"k":int(rng.integers(1,201)),"bm25_weight":float(rng.random())}; key=(c['k'],c['bm25_weight'])
  if key not in seen: seen.add(key); rows.append({"candidate":c,"dev":ev(c,'dev')})
 result={"seed":a.seed,"draws":rows,"selected":choose(rows)}; result["selected"]["public_test_after_selection"]=ev(result["selected"]["candidate"],'test')
a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)+"\n")
