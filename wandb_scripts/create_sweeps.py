"""Create wandb sweeps for the HPO (param-tuning) and 10-fold phases and save their IDs.

Reads each listed YAML, calls wandb.sweep(), and stores
friendly_name -> sweep_id under named top-level groups in
wandb_scripts/sweep_ids.yaml (e.g. ``hpo_rq1``, ``folds_rq1``). The IDs
file is rewritten after every successful sweep creation so a crash
mid-run does not lose progress. Existing entries are preserved; any
sweep already present in the file is skipped, so the script is safe to
re-run.

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

HPO_RQ1_CV3 = {
    "all":         "sweeps/hpo_kge_transd_cv3.yml",
    "only_pheno":  "sweeps/hpo_kge_transd_no_func_no_site_cv3.yml",
    "no_site":     "sweeps/hpo_kge_transd_no_site_cv3.yml",
    "no_function": "sweeps/hpo_kge_transd_no_func_cv3.yml",
    "indigena":    "sweeps/hpo_kge_transd_indigena_cv3.yml",
}

HPO_RQ1_CV3_V2 = {
    "all":         "sweeps/hpo_kge_transd_cv3_v2.yml",
    "only_pheno":  "sweeps/hpo_kge_transd_no_func_no_site_cv3_v2.yml",
    "no_site":     "sweeps/hpo_kge_transd_no_site_cv3_v2.yml",
    "no_function": "sweeps/hpo_kge_transd_no_func_cv3_v2.yml",
    "indigena":    "sweeps/hpo_kge_transd_indigena_cv3_v2.yml",
}

FOLDS_RQ1_CV3_BS32K = {
    "all_owl2vecstar":         "sweeps/hpo_kge_transd_folds_owl2vecstar.yml",
    "all_gda":                 "sweeps/hpo_kge_transd_folds_gda.yml",
    "only_pheno_owl2vecstar":  "sweeps/hpo_kge_transd_no_func_no_site_folds_owl2vecstar.yml",
    "only_pheno_gda":          "sweeps/hpo_kge_transd_no_func_no_site_folds_gda.yml",
    "no_site_owl2vecstar":     "sweeps/hpo_kge_transd_no_site_folds_owl2vecstar.yml",
    "no_site_gda":             "sweeps/hpo_kge_transd_no_site_folds_gda.yml",
    "no_function_owl2vecstar": "sweeps/hpo_kge_transd_no_func_folds_owl2vecstar.yml",
    "no_function_gda":         "sweeps/hpo_kge_transd_no_func_folds_gda.yml",
    "indigena_owl2vecstar":    "sweeps/hpo_kge_transd_indigena_folds_owl2vecstar.yml",
    "indigena_gda":            "sweeps/hpo_kge_transd_indigena_folds_gda.yml",
}

SWEEP_GROUPS = {
    "hpo_rq1":             HPO_RQ1,
    "hpo_rq1_cv3":         HPO_RQ1_CV3,
    "hpo_rq1_cv3_v2":      HPO_RQ1_CV3_V2,
    "folds_rq1_cv3_bs32k": FOLDS_RQ1_CV3_BS32K,
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

    created = 0
    for group_key, group_map in SWEEP_GROUPS.items():
        group = ids.setdefault(group_key, {})
        for name, rel_path in group_map.items():
            if name in group:
                print(f"SKIP {group_key}/{name}: already has id {group[name]}")
                continue
            yml_path = REPO_ROOT / rel_path
            if not yml_path.exists():
                print(f"SKIP {group_key}/{name}: {rel_path} not found", file=sys.stderr)
                continue
            sweep_cfg = yaml.safe_load(yml_path.read_text())
            sweep_id = wandb.sweep(sweep_cfg, entity=entity, project=project)
            print(f"{group_key}/{name}: {sweep_id}  ({rel_path})")
            group[name] = sweep_id
            save_ids(ids)
            created += 1

    print(f"\nCreated {created} new sweep(s); IDs saved to {IDS_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
