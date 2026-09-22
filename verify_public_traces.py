"""Verify archived six-call decisions without redistributing NFCorpus data.

This checks trace integrity and selection logic. It does not recompute the
development or public-test metrics; use the pinned upstream dataset for that.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRESH = ROOT / "fresh-repeat-v5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    freeze = json.loads((FRESH / "selection_lock.json").read_text())
    require(freeze["status"] == "FROZEN_BEFORE_NEW_TEST_EVALUATION", "selection was not locked")
    require(len(freeze["trajectories"]) == 3, "expected three fresh trajectories")
    chosen = json.loads((FRESH / "frozen_unique_candidate.json").read_text())
    omitted_inputs = []
    trajectories = []
    for item in freeze["trajectories"]:
        directory = FRESH / item["name"]
        require(sha(directory / "state.json") == item["state_sha256"], f"{item['name']}: state hash")
        require(sha(directory / "TASK.md") == item["task_sha256"], f"{item['name']}: task hash")
        for name, expected in item["trial_file_sha256"].items():
            require(sha(directory / name) == expected, f"{item['name']}: trial file {name}")
        for name, expected in item["immutable_input_sha256"].items():
            if name.startswith("data/"):
                omitted_inputs.append(f"{item['name']}/{name}")
                continue
            # The three agents used a copied evaluator identical to the root evaluator.
            public_copy = ROOT / "evaluator" / Path(name).name
            require(sha(public_copy) == expected, f"{item['name']}: evaluator hash {name}")
        trials = json.loads((directory / "state.json").read_text())["trials"]
        require(len(trials) == 6 and all(row["status"] == "SUCCESS" for row in trials), f"{item['name']}: budget")
        require(len({json.dumps(row["candidate"], sort_keys=True) for row in trials}) == 6, f"{item['name']}: repeated candidate")
        for row in trials:
            hypothesis = directory / f"runs/trial-{row['trial']:02d}/hypothesis.txt"
            require(len(hypothesis.read_text().strip()) >= 20, f"{item['name']}: missing prior hypothesis")
        winner = max(trials, key=lambda row: (row["metrics"]["ndcg@10"], -row["trial"]))
        require(winner["trial"] == item["selected_trial"], f"{item['name']}: selected trial")
        require(winner["candidate"] == item["selected_candidate"] == chosen, f"{item['name']}: chosen candidate")
        require(winner["metrics"]["ndcg@10"] == item["selected_dev_ndcg_exp_at_10"], f"{item['name']}: development feedback")
        trajectories.append({"name": item["name"], "calls": len(trials), "selected_trial": winner["trial"], "candidate": chosen})
    test = json.loads((FRESH / "public_test_selected.json").read_text())
    require(test["test_labels_read"] is True and test["candidate"] == chosen and test["query_count"] == 323, "public-test metadata")
    matrix = json.loads((ROOT / "results/matrix.json").read_text())
    match = [row for row in matrix if row["candidate"] == chosen]
    require(len(match) == 1 and abs(match[0]["test_exp"] - test["metrics"]["nDCG_exp@10"]) < 1e-14, "archived matrix reference")
    report = {
        "status": "PASS_TRACE_ONLY",
        "trajectories": trajectories,
        "selected_public_test_ndcg_exp_at_10": test["metrics"]["nDCG_exp@10"],
        "omitted_derived_data_inputs": omitted_inputs,
        "scope": "Checks archived hashes, hypotheses, six-call budgets and development selection. Dataset/cache-dependent metrics require local reconstruction from upstream NFCorpus; source text and qrels are not redistributed here.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
