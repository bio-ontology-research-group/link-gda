import wandb
import tomllib

# Initialize the W&B API
api = wandb.Api()

with open("../config.toml", "rb") as f:
    config = tomllib.load(f)

entity = config["wandb"]["entity"]
project = config["wandb"]["project"]

PROJECTOR_PARAM = "projector_name"
RUNS_PER_PROJECTOR = 10
EXPECTED_TOTAL_RUNS = 20  # 2 projectors x 10 runs each

GGDA_SWEEPS = {
    "indigena": "fgpk2bwd",
    "only_pheno":  "1c4xhafc",
    "no_pheno":    "s36n5o98", #"b6rmb691",
    "no_site":     "vts5rhbc",# "hpi6u25o",
    "no_function": "1wtlgkqr", #"1jm285xu",
    "all":         "34sda2me" #"2zb031rp"
}



METRICS = [
    "test_imac_bma_mr",
    "test_imac_bma_mrr",
    "test_imac_bma_hits@1",
    "test_imac_bma_hits@3",
    "test_imac_bma_hits@10",
    "test_imac_bma_hits@100",
    "test_imac_bma_auc",
    "test_imac_bmm_mr",
    "test_imac_bmm_mrr",
    "test_imac_bmm_hits@1",
    "test_imac_bmm_hits@3",
    "test_imac_bmm_hits@10",
    "test_imac_bmm_hits@100",
    "test_imac_bmm_auc",
]


def get_mean_and_std_per_projector(sweep_id, metrics):
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # projector_name -> {metric -> [values]}
    projector_data = {}
    total_runs = 0

    for run in sweep.runs:
        if run.state != "finished":
            continue
        projector = run.config.get(PROJECTOR_PARAM, "unknown")
        if projector not in projector_data:
            projector_data[projector] = {metric: [] for metric in metrics}
        total_runs += 1
        for metric in metrics:
            value = run.summary.get(metric)
            if value is not None:
                projector_data[projector][metric].append(value)

    if total_runs != EXPECTED_TOTAL_RUNS:
        print(f"  WARNING: expected {EXPECTED_TOTAL_RUNS} finished runs, got {total_runs}.")

    for projector, counts in projector_data.items():
        first_metric = metrics[0]
        n = len(counts[first_metric])
        if n != RUNS_PER_PROJECTOR:
            print(f"  WARNING: projector '{projector}' has {n}/{RUNS_PER_PROJECTOR} runs.")

    results = {}
    for projector, metric_values in projector_data.items():
        stats = {}
        for metric, values in metric_values.items():
            if not values:
                continue
            mean = sum(values) / len(values)
            std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
            stats[metric] = {"mean": mean, "std": std}
        results[projector] = stats

    return results


def print_as_tex(stats, metrics):
    string = ""
    for metric in metrics:
        if metric not in stats:
            string += "N/A & "
            continue
        string += f"{stats[metric]['mean']:.2f}\\std{{{stats[metric]['std']:.2f}}} & "
    string = string[:-2] + " \\\\"
    print(string)


if __name__ == "__main__":
    metric_groups = {
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

    for sweep_name, sweep_id in GGDA_SWEEPS.items():
        print(f"\n=== {sweep_name} (sweep: {sweep_id}) ===")
        per_projector = get_mean_and_std_per_projector(sweep_id, METRICS)

        for group_name, group_metrics in metric_groups.items():
            print(f"\n  -- {group_name} --")
            for projector, stats in per_projector.items():
                print(f"  [{projector}]")
                print_as_tex(stats, group_metrics)

        print("----------------------------------------")
