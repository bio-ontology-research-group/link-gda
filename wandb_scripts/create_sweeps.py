"""Create wandb sweeps for the HPO (param-tuning) phase and save their IDs.

Reads each listed YAML, calls wandb.sweep(), and stores
friendly_name -> sweep_id under the ``hpo_rq1`` key in
wandb_scripts/sweep_ids.yaml. The IDs file is rewritten after every
successful sweep creation so a crash mid-run does not lose progress.

Run from the repository root:

    python wandb_scripts/create_sweeps.py
"""

import sys
import tomllib
from pathlib import Path

import yaml
import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
IDS_FILE = REPO_ROOT / "wandb_scripts" / "sweep_ids.yaml"

HPO_RQ1 = {
    "all":         "sweeps/hpo_kge_transd.yml",
    "only_pheno":  "sweeps/hpo_kge_transd_no_func_no_site.yml",
    "no_site":     "sweeps/hpo_kge_transd_no_site.yml",
    "no_function": "sweeps/hpo_kge_transd_no_func.yml",
    "indigena":    "sweeps/hpo_kge_transd_indigena.yml",
}


def load_ids():
    if IDS_FILE.exists():
        return yaml.safe_load(IDS_FILE.read_text()) or {}
    return {}


def save_ids(ids):
    IDS_FILE.write_text(yaml.safe_dump(ids, sort_keys=False))


def main():
    with open(REPO_ROOT / "config.toml", "rb") as f:
        config = tomllib.load(f)
    entity = config["wandb"]["entity"]
    project = config["wandb"]["project"]

    ids = load_ids()
    hpo_rq1 = ids.setdefault("hpo_rq1", {})

    for name, rel_path in HPO_RQ1.items():
        yml_path = REPO_ROOT / rel_path
        if not yml_path.exists():
            print(f"SKIP {name}: {rel_path} not found", file=sys.stderr)
            continue
        sweep_cfg = yaml.safe_load(yml_path.read_text())
        sweep_id = wandb.sweep(sweep_cfg, entity=entity, project=project)
        print(f"{name}: {sweep_id}  ({rel_path})")
        hpo_rq1[name] = sweep_id
        save_ids(ids)

    print(
        f"\nSaved {len(hpo_rq1)} sweep ID(s) under hpo_rq1 in "
        f"{IDS_FILE.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
