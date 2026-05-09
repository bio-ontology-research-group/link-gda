"""Pick the best HPO config per projector by mean metric across CV folds.

For each sweep listed under ``hpo_rq1_cv3``, groups finished runs by
(projector_name, batch_size, embedding_dim, learning_rate), averages the
selection metric across folds in the group, and prints the best
(lowest-mean for MR, highest for MRR) combination per projector.

Run from the repository root:
    python wandb_scripts/best_config_cv.py
"""

from collections import defaultdict
from pathlib import Path
import statistics

import wandb
import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_IDS_FILE = Path(__file__).resolve().parent / "sweep_ids.yaml"
SWEEP_IDS_KEY = "hpo_rq1_cv3"

METRIC = "test_imac_bma_mr"
CRITERION = "minimize"
EXPECTED_FOLDS = 3
PARAMETERS = ["batch_size", "embedding_dim", "learning_rate"]


def main():
    with open(REPO_ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    entity, project = cfg["wandb"]["entity"], cfg["wandb"]["project"]

    sweeps = yaml.safe_load(SWEEP_IDS_FILE.read_text()).get(SWEEP_IDS_KEY) or {}
    if not sweeps:
        print(f"No sweeps under '{SWEEP_IDS_KEY}'. Run create_sweeps.py first.")
        return

    api = wandb.Api()

    for name, sid in sweeps.items():
        print(f"\n=== {name} (sweep: {sid}) ===")
        sweep = api.sweep(f"{entity}/{project}/{sid}")

        # group_key -> list of (fold, value)
        groups = defaultdict(list)
        for run in sweep.runs:
            if run.state != "finished":
                continue
            value = run.summary.get(METRIC)
            if value is None:
                continue
            proj = run.config.get("projector_name", "unknown")
            params = tuple(run.config.get(p) for p in PARAMETERS)
            fold = run.config.get("fold")
            groups[(proj,) + params].append((fold, value))

        if not groups:
            print(f"  No finished runs with metric '{METRIC}'.")
            continue

        # Aggregate per group
        per_proj_best = {}
        for key, fold_values in groups.items():
            proj = key[0]
            vals = [v for _, v in fold_values]
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            n = len(vals)
            entry = (mean, std, n, key, fold_values)
            cur = per_proj_best.get(proj)
            if cur is None:
                per_proj_best[proj] = entry
            else:
                if CRITERION == "minimize" and mean < cur[0]:
                    per_proj_best[proj] = entry
                elif CRITERION == "maximize" and mean > cur[0]:
                    per_proj_best[proj] = entry

        for proj, (mean, std, n, key, fold_values) in per_proj_best.items():
            params = dict(zip(PARAMETERS, key[1:]))
            print(f"\n  -- Projector: {proj} --")
            print(f"  Best mean {METRIC} = {mean:.4f} +/- {std:.4f}  (n={n}/{EXPECTED_FOLDS} folds)")
            print(f"  Per-fold values: {sorted(fold_values)}")
            print(f"  Parameters:")
            for p, v in params.items():
                print(f"    {p}: {v}")
            if n != EXPECTED_FOLDS:
                print(f"  WARNING: expected {EXPECTED_FOLDS} folds, got {n}")


if __name__ == "__main__":
    main()
