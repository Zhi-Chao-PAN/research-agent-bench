import argparse, json
from pathlib import Path
from core import evaluate
p=argparse.ArgumentParser(); p.add_argument('--candidate', type=Path, required=True); p.add_argument('--cache', type=Path, required=True); p.add_argument('--qrels', type=Path, required=True); p.add_argument('--out', type=Path, required=True); a=p.parse_args()
r={"candidate":json.loads(a.candidate.read_text()), "metrics":evaluate(a.cache,a.qrels,json.loads(a.candidate.read_text())), "test_labels_read":False}; a.out.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2))
