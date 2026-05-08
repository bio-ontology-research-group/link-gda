"""Aggregate 10-fold metrics per (setting, projector) across base + retry sweeps.

For each setting key (e.g. "all_gda", "no_site_owl2vecstar"), reads finished
runs from both the original `folds_rq1` sweep and the `folds_rq1_retry`
sweep (if any), dedupes by `fold` (the original wins on tie), and prints
mean +/- std across the 10 folds for each metric group.

Run from the repository root:
    python wandb_scripts/extract_metrics_from_folds.py
"""

import tomllib
from pathlib import Path

import wandb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IDS_FILE = REPO_ROOT / "wandb_scripts" / "sweep_ids.yaml"

EXPECTED_FOLDS = 10

METRIC_GROUPS = {
    "Inductive BMA": [
        "test_imac_bma_mr", "test_imac_bma_mrr",
        "test_imac_bma_hits@1", "test_imac_bma_hits@3",
        "test_imac_bma_hits@10", "test_imac_bma_hits@100",
        "test_imac_bma_auc",
    ],
    "Inductive BMM": [
        "test_imac_bmm_mr", "test_imac_bmm_mrr",
        "test_imac_bmm_hits@1", "test_imac_bmm_hits@3",
        "test_imac_bmm_hits@10", "test_imac_bmm_hits@100",
        "test_imac_bmm_auc",
    ],
}


def collect_runs(api, entity, project, sweep_id):
    """Return {fold: {metric: value}} for finished runs in a sweep."""
    by_fold = {}
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    for run in sweep.runs:
        if run.state != "finished":
            continue
        fold = run.config.get("fold")
        if fold is None:
            continue
        by_fold[int(fold)] = dict(run.summary)
    return by_fold


def stats(values):
    n = len(values)
    mean = sum(values) / n
    std = (sum((x - mean) ** 2 for x in values) / n) ** 0.5
    return mean, std


def print_as_tex(per_metric_stats, metrics):
    parts = []
    for m in metrics:
        if m in per_metric_stats:
            mean, std = per_metric_stats[m]
            parts.append(f"{mean:.2f}\\std{{{std:.2f}}}")
        else:
            parts.append("N/A")
    print(" & ".join(parts) + " \\\\")


def main():
    with open(REPO_ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    entity, project = cfg["wandb"]["entity"], cfg["wandb"]["project"]

    ids = yaml.safe_load(IDS_FILE.read_text()) or {}
    base = ids.get("folds_rq1", {})
    retry = ids.get("folds_rq1_retry", {})

    api = wandb.Api()

    all_metrics = sorted({m for ms in METRIC_GROUPS.values() for m in ms})

    for name, base_id in base.items():
        merged = collect_runs(api, entity, project, base_id)
        retry_id = retry.get(name)
        if retry_id:
            for fold, summary in collect_runs(api, entity, project, retry_id).items():
                merged.setdefault(fold, summary)

        n = len(merged)
        missing = sorted(set(range(EXPECTED_FOLDS)) - set(merged))
        suffix = "" if retry_id is None else f" + {retry_id}"
        print(f"\n=== {name} (sweep: {base_id}{suffix}) ===")
        print(f"  finished folds: {n}/{EXPECTED_FOLDS}; missing: {missing}")

        if n == 0:
            continue

        per_metric_stats = {}
        for m in all_metrics:
            values = [s[m] for s in merged.values() if m in s and s[m] is not None]
            if values:
                per_metric_stats[m] = stats(values)

        for group_name, group_metrics in METRIC_GROUPS.items():
            print(f"  -- {group_name} --")
            print_as_tex(per_metric_stats, group_metrics)


if __name__ == "__main__":
    main()
