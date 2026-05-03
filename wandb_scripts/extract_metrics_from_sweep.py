import wandb
import tomli
# Initialize the W&B API
api = wandb.Api()

with open("config.toml", "rb") as f:
    config = tomli.load(f)

entity = config["wandb"]["entity"]
project = config["wandb"]["project"]
 


def get_mean_and_std(sweep_id, metrics):
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    metrics_summary = {metric: [] for metric in metrics}

    try:
        for run in sweep.runs:
            run_data = run.summary
            for metric in metrics:
                metrics_summary[metric].append(run_data[metric])

        metrics_stats = {}
        for metric, values in metrics_summary.items():
            mean_value = sum(values) / len(values)
            std_value = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5
            metrics_stats[metric] = {'mean': mean_value, 'std': std_value}

        return metrics_stats
    except Exception as e:
        print(f"An error occurred while processing sweep {sweep_id}: {e}")
        return None

def print_as_tex(stats):
    string = ""
    for metric, stat in stats.items():
        string += f"{stat['mean']:.2f}\std{{{stat['std']:.2f}}} & "
    string = string[:-2] + " \\\\"
    print(string)
        
if __name__ == "__main__":

                                                                                                                  

    sweeps_gene_reconstruction = { "transd_graph4" : "xn7jqdr0",
                                   "convkb_d_graph4_repr" : "eg07b33e",
                                   "convkb_d_graph4" : "tgynx701",
                                     }

    sweeps = sweeps_gene_reconstruction
    for name, sweep_id in sweeps.items():
        # print(f"Metrics for {name} (Sweep ID: {sweep_id}):")
        # metrics_to_extract = ["mac_mr", "mac_mrr", "mac_hits@1", "mac_hits@3", "mac_hits@10", "mac_hits@100", "mac_auc"]
        
        # stats = get_mean_and_std(sweep_id, metrics_to_extract)
        # print_as_tex(stats)
        # for metric, stat in stats.items():
            # print(f"{metric}: Mean = {stat['mean']}, Std = {stat['std']}")

        # continue

        print(f"Inductive BMA metrics for {name} (Sweep ID: {sweep_id}):")
        inductive_bma_metrics_to_extract = ["test_imac_bma_mr", "test_imac_bma_mrr", "test_imac_bma_hits@100", "test_imac_bma_auc"]
        # inductive_metrics_to_extract = ["test_imac_bma_mr", "test_imac_bma_mrr", "test_imac_bma_hits@1", "test_imac_bma_hits@3", "test_imac_bma_hits@10", "test_imac_bma_hits@100", "test_imac_bma_auc"]
        inductive_bma_stats = get_mean_and_std(sweep_id, inductive_bma_metrics_to_extract)
        print_as_tex(inductive_bma_stats)

        inductive_bmm_metrics_to_extract = ["test_imac_bmm_mr", "test_imac_bmm_mrr", "test_imac_bmm_hits@100", "test_imac_bmm_auc"]
        print(f"Inductive BMM metrics for {name} (Sweep ID: {sweep_id}):")
        inductive_bmm_stats = get_mean_and_std(sweep_id, inductive_bmm_metrics_to_extract)
        print_as_tex(inductive_bmm_stats)
        
        print(f"Transductive similarity metrics for {name} (Sweep ID: {sweep_id}):")
        transductive_sim_metrics_to_extract = ["test_sim_tmac_mr", "test_sim_tmac_mrr", "test_sim_tmac_hits@100", "test_sim_tmac_auc"]
        transductive_sim_stats = get_mean_and_std(sweep_id, transductive_sim_metrics_to_extract)
        if transductive_sim_stats:
            print_as_tex(transductive_sim_stats)
        else:
            print("No transductive similarity metrics found.")
            
        print(f"Transductive function metrics for {name} (Sweep ID: {sweep_id}):")
        transductive_func_metrics_to_extract = ["test_func_tmac_mr", "test_func_tmac_mrr", "test_func_tmac_hits@100", "test_func_tmac_auc"]
        transductive_func_stats = get_mean_and_std(sweep_id, transductive_func_metrics_to_extract)
        if transductive_func_stats:
            print_as_tex(transductive_func_stats)
        else:
            print("No transductive function metrics found.")

        print("----------------------------------------")
        if transductive_func_stats:
            print_as_tex(transductive_func_stats)
        if transductive_sim_stats:
            print_as_tex(transductive_sim_stats)
        print_as_tex(inductive_bmm_stats)
        print_as_tex(inductive_bma_stats)
        print("----------------------------------------")
