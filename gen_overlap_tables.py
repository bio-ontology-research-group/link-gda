"""Emit LaTeX rows for the overlap-stratified tables from the average-rank results.

Single source of numbers so the main stratified table and the supplement full table
stay consistent. The numbers used to be a DATA dict pasted in from two log files; they
are now read from the strata TSV that strata_from_labels.py writes, so the table and
the per-instance score files cannot drift apart. Values are (mean, sd) across the ten
folds with average-rank tie handling, exactly as before.

Two things changed with the source and are worth stating rather than discovering later:

The overlap definition is now the graph_dump one, which counts only the causes_phenotype
edges the training graph actually carries (leakage_overlap.py, --overlap-source
train-split). The retired DATA dict used the full-train definition, which also counts the
validation-holdout diseases' edges. Pass --table-file .../strata_all_methods_train_csv.tsv
to reproduce the old stratification.

The retired DATA dict held UNCALIBRATED numbers, while the project reports calibrated as
the headline setting. --setting therefore defaults to calibrated and must be set to raw
to reproduce the old rows.

The retired dict also mixed projections: LinkGDA rows came from GDAProjector and INDIGENA
from OWL2Vec*. --projection sets one for all rows; --override reinstates a per-method
choice, e.g. --override INDIGENA=owl2vecstar.

    python gen_overlap_tables.py
    python gen_overlap_tables.py --setting raw --override INDIGENA=owl2vecstar
"""
import csv

import click as ck

METRICS = ["MR", "MRR", "H@1", "H@3", "H@10", "H@100", "AUC"]
STRATA = ["zero", "some", "all"]
STRATUM_ROWS = {"zero": "zero overlap", "some": "some overlap", "all": "all"}

# Display name -> method key in the strata TSV; {projection} is filled from --projection.
ROWS = [
    ("SimGIC", "SimGIC"),
    ("Resnik-BMA", "Resnik-BMA"),
    ("Resnik-BMM", "Resnik-BMM"),
    ("Lin-BMA", "Lin-BMA"),
    ("Lin-BMM", "Lin-BMM"),
    ("Exomiser-Phive", "Exomiser-Phive"),
    ("INDIGENA", "INDIGENA_{projection}"),
    ("ULTRA-pfs", "ULTRA-pfs-zeroshot_{projection}"),
    ("\\method{}-p", "LinkGDA-p_{projection}"),
    ("\\method{}-ps", "LinkGDA-ps_{projection}"),
    ("\\method{}-pf", "LinkGDA-pf_{projection}"),
    ("\\method{}-pfs", "LinkGDA-pfs_{projection}"),
]


def load_data(path, setting, projection, overrides):
    """DATA[display name][stratum][metric] = (mean, sd), read from a strata TSV."""
    table = {}
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["setting"] != setting:
                continue
            table[(row["method"], row["stratum"])] = row

    data, order = {}, []
    for display, key in ROWS:
        resolved = key.format(projection=overrides.get(display, projection))
        by_stratum = {}
        for short, stratum in STRATUM_ROWS.items():
            row = table.get((resolved, stratum))
            if row is None:
                break
            by_stratum[short] = {m: (float(row[f"{m}_mean"]), float(row[f"{m}_sd"])) for m in METRICS}
        if len(by_stratum) == len(STRATUM_ROWS):
            data[display] = by_stratum
            order.append(display)
    if not data:
        raise SystemExit(f"no rows for setting={setting} projection={projection} in {path}")
    return data, order


def fmt(metric, mean, std):
    return f"{mean:.2f}\\std{{{std:.2f}}}"


def best(data, order, metric, stratum):
    """method with best (min MR / max otherwise) mean in this stratum."""
    vals = {m: data[m][stratum][metric][0] for m in order}
    return min(vals, key=vals.get) if metric == "MR" else max(vals, key=vals.get)


def cell(data, order, method, stratum, metric):
    mean, std = data[method][stratum][metric]
    s = fmt(metric, mean, std)
    return f"\\f{{{s}}}" if best(data, order, metric, stratum) == method else s


def main_table(data, order, metrics=("MR", "MRR", "H@10")):
    print("% ---- MAIN stratified table (metrics: " + ",".join(metrics) + ") ----")
    for m in order:
        cells = " & ".join(cell(data, order, m, st, me) for st in STRATA for me in metrics)
        print(f"    {m:<14} & {cells} \\\\")
        if m == "INDIGENA":
            print("    \\midrule")


def supp_table(data, order):
    print("\n% ---- SUPPLEMENT full table (all metrics, stratum = all) ----")
    for m in order:
        cells = " & ".join(cell(data, order, m, "all", me) for me in METRICS)
        print(f"    {m:<14} & {cells} \\\\")
        if m == "INDIGENA":
            print("    \\midrule")


@ck.command()
@ck.option("--table-file", default="data/results/strata_all_methods_graph_dump.tsv", show_default=True)
@ck.option("--setting", type=ck.Choice(["calibrated", "raw"]), default="calibrated", show_default=True)
@ck.option("--projection", default="owl2vecstar_gda", show_default=True)
@ck.option("--override", multiple=True, help="Per-row projection as 'display name=projection', repeatable")
def main(table_file, setting, projection, override):
    overrides = dict(item.split("=", 1) for item in override)
    data, order = load_data(table_file, setting, projection, overrides)
    print(f"% source: {table_file}  setting: {setting}  projection: {projection}"
          + (f"  overrides: {overrides}" if overrides else ""))
    main_table(data, order)
    supp_table(data, order)


if __name__ == "__main__":
    main()
