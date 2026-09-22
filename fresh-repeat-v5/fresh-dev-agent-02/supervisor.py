"""Six-call development-only gate. The agent's proposal file is candidate JSON only."""
import argparse, fcntl, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent; FROZEN=ROOT.parent / "evaluator"
p=argparse.ArgumentParser(); p.add_argument("--candidate",type=Path,required=True); p.add_argument("--hypothesis",required=True); a=p.parse_args(); path=a.candidate.resolve()
if path.parent != ROOT / "candidates": raise ValueError("candidate must be directly in candidates/")
c=json.loads(path.read_text()); sys.path.insert(0,str(FROZEN)); from core import validate, sha256; validate(c)
if len(a.hypothesis.strip()) < 20: raise ValueError("hypothesis must contain at least 20 non-space characters")
protocol=json.loads((ROOT/"TASK.md").read_text().split("```json\n")[1].split("\n```")[0])
with (ROOT/"supervisor.lock").open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX); state_path=ROOT/"state.json"; state=json.loads(state_path.read_text()) if state_path.exists() else {"started_at":time.time(),"trials":[]}
 if len(state["trials"])>=6 or time.time()-state["started_at"]>=900: raise RuntimeError("six-trial or 15-minute budget exhausted")
 if any(t["candidate"]==c for t in state["trials"]): raise ValueError("duplicate candidate")
 for name,digest in protocol["immutable_sha256"].items():
  if sha256(ROOT/name)!=digest: raise RuntimeError("fixed input changed: "+name)
 n=len(state["trials"])+1; out=ROOT/"runs"/f"trial-{n:02d}"; out.mkdir(parents=True); (out/"candidate.json").write_bytes(path.read_bytes())
 record={"trial":n,"candidate":c,"hypothesis":a.hypothesis,"status":"RUNNING"}; (out/"hypothesis.txt").write_text(a.hypothesis+"\n"); state["trials"].append(record); state_path.write_text(json.dumps(state,indent=2)+"\n")
 cmd=[sys.executable,str(FROZEN/"evaluate_candidate.py"),"--candidate",str(out/"candidate.json"),"--cache",str(ROOT/"data/dev_ranks.npz"),"--qrels",str(ROOT/"data/dev_qrels.tsv"),"--out",str(out/"result.json")]
 started=time.perf_counter()
 try:
  remaining=max(1, protocol["max_wall_seconds"]-(time.time()-state["started_at"])); z=subprocess.run(cmd,capture_output=True,text=True,timeout=min(protocol["per_trial_timeout_seconds"],remaining)); (out/"stdout.log").write_text(z.stdout); (out/"stderr.log").write_text(z.stderr); record["status"]="SUCCESS" if z.returncode==0 else "FAILED"; record["exit_code"]=z.returncode
  if z.returncode==0: record["metrics"]=json.loads((out/"result.json").read_text())["metrics"]
 except subprocess.TimeoutExpired: record["status"]="TIMEOUT"
 record["wall_seconds"]=time.perf_counter()-started; state_path.write_text(json.dumps(state,indent=2)+"\n"); print(json.dumps(record,indent=2))
