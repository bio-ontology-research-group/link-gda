import wandb
import tomllib

# Initialize the W&B API
api = wandb.Api()

with open("../config.toml", "rb") as f:
    config = tomllib.load(f)

entity = config["wandb"]["entity"]
project = config["wandb"]["project"]

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECTOR_PARAM = "projector_name"  # config key used to identify the projector
EXPECTED_RUNS = 36  # minimum number of finished runs that must have reported the metric (total, across all projectors)

SWEEPS = {
    "only_pheno":  "12kx2a1c",
    "no_pheno":    "lmpbjjg1",
    "no_site":     "r4siihd1",
    "no_function": "xhdc75ib",
    "all":         "fb7e9wrk"
}

INDIGENA_SWEEPS = {
    "only_pheno": "9kk4wy80",
    "no_site": "s1vhm4a2",
    "no_function": "qasqxfjx",
    "all": "db8np4wh"
}

METRIC = "test_imac_bma_mr"
CRITERION = "minimize"  # "maximize" or "minimize"

PARAMETERS_TO_RETRIEVE = [
    "learning_rate",
    "batch_size",
    "embedding_dim",
    "projector_name",
]
# ──────────────────────────────────────────────────────────────────────────────


def get_best_config(sweep_id, metric, criterion, parameters, expected_runs=None):
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Group runs by projector
    projector_best = {}   # projector_name -> (best_value, best_run)
    valid_runs = 0

    for run in sweep.runs:
        if run.state != "finished":
            continue
        value = run.summary.get(metric)
        if value is None:
            continue

        valid_runs += 1
        projector = run.config.get(PROJECTOR_PARAM, "unknown")

        if projector not in projector_best:
            projector_best[projector] = (value, run)
        else:
            current_best, _ = projector_best[projector]
            if criterion == "maximize" and value > current_best:
                projector_best[projector] = (value, run)
            elif criterion == "minimize" and value < current_best:
                projector_best[projector] = (value, run)

    if expected_runs is not None and valid_runs != expected_runs:
        print(f"  WARNING: only {valid_runs}/{expected_runs} finished runs reported '{metric}'.")

    if not projector_best:
        print(f"No finished runs with metric '{metric}' found in sweep {sweep_id}.")
        return None

    print(f"  Finished runs with metric: {valid_runs}")

    results = {}
    for projector, (best_value, best_run) in projector_best.items():
        print(f"\n  -- Projector: {projector} --")
        print(f"  Best run: {best_run.name} (id: {best_run.id})")
        print(f"  {metric} = {best_value}")
        print(f"  Parameters:")
        result = {}
        for param in parameters:
            value = best_run.config.get(param)
            result[param] = value
            print(f"    {param}: {value}")
        results[projector] = result

    return results


if __name__ == "__main__":
    SWEEPS_TO_USE = INDIGENA_SWEEPS
    for name, sweep_id in SWEEPS_TO_USE.items():
        print(f"\n=== {name} (sweep: {sweep_id}) ===")
        get_best_config(sweep_id, METRIC, CRITERION, PARAMETERS_TO_RETRIEVE, EXPECTED_RUNS)
        print("----------------------------------------")
