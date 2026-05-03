"""
Evaluate Exomiser prioritisers on the same test folds as the KGE model.

Runs three phenotype-only prioritisers:
  - hiPhive (cross-species + PPI)
  - Phive (cross-species, no PPI)
  - PhenIX (human HPO only)

Outputs TSV files in the same format as kge_transd.py evaluation,
then computes metrics using the shared compute_metrics function.
"""

import os
import glob
import jpype

# --- JVM setup (following the pattern in kge_transd.py) ---
exomiser_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "exomiser", "exomiser-cli-14.0.0")
exomiser_jars = glob.glob(os.path.join(exomiser_dir, "lib", "*.jar"))
exomiser_jars.append(os.path.join(exomiser_dir, "exomiser-cli-14.0.0.jar"))

phenotype_data_dir = os.path.join(exomiser_dir, "data", "2406_phenotype")

jpype.startJVM(
    jpype.getDefaultJVMPath(),
    "-ea",
    "-Xmx10g",
    classpath=exomiser_jars,
    convertStrings=True,
)

import jpype.imports  # noqa: E402
from java.nio.file import Paths as JPaths

# H2 DataSource
from com.zaxxer.hikari import HikariDataSource

# Exomiser DAOs
from org.monarchinitiative.exomiser.core.phenotype.dao import (
    HumanPhenotypeOntologyDao,
    MousePhenotypeOntologyDao,
    ZebraFishPhenotypeOntologyDao,
)

# Exomiser services
from org.monarchinitiative.exomiser.core.phenotype.service import OntologyServiceImpl
from org.monarchinitiative.exomiser.core.phenotype import PhenotypeMatchService
from org.monarchinitiative.exomiser.core.prioritisers.service import (
    ModelServiceImpl,
    PriorityService,
)
from org.monarchinitiative.exomiser.core.prioritisers.dao import DefaultDiseaseDao
from org.monarchinitiative.exomiser.core.prioritisers import (
    PriorityFactoryImpl,
    HiPhiveOptions,
)
from org.monarchinitiative.exomiser.core.prioritisers.util import DataMatrixIO

# Exomiser model
from org.monarchinitiative.exomiser.core.model import Gene
from java.util import ArrayList
from java.util.stream import Collectors

import click as ck
import numpy as np
import pandas as pd
from evaluate_sem_sim import compute_metrics, print_as_tex
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def uri_to_hpo(uri):
    """'http://purl.obolibrary.org/obo/HP_0009102' -> 'HP:0009102'"""
    return uri.split("/")[-1].replace("_", ":")


def uri_to_entrez(uri):
    """'http://mowl.borg/51733' -> 51733"""
    return int(uri.split("/")[-1])


def build_priority_factory():
    """Manually wire Exomiser phenotype components (no Spring Boot needed)."""
    db_path = os.path.join(phenotype_data_dir, "2406_phenotype")
    jdbc_url = (f"jdbc:h2:file:{db_path}"
                ";ACCESS_MODE_DATA=r"
                ";INIT=SET SCHEMA EXOMISER")

    ds = HikariDataSource()
    ds.setJdbcUrl(jdbc_url)
    ds.setUsername("sa")
    ds.setPassword("")

    # DAOs (all backed by the H2 phenotype database)
    human_dao = HumanPhenotypeOntologyDao(ds)
    mouse_dao = MousePhenotypeOntologyDao(ds)
    fish_dao = ZebraFishPhenotypeOntologyDao(ds)

    # Services
    ontology_service = OntologyServiceImpl(human_dao, mouse_dao, fish_dao)
    phenotype_match_service = PhenotypeMatchService(ontology_service)
    model_service = ModelServiceImpl(ds)
    disease_dao = DefaultDiseaseDao(ds)
    priority_service = PriorityService(model_service, phenotype_match_service,
                                       disease_dao)

    # PPI random-walk matrix
    rw_path = JPaths.get(os.path.join(phenotype_data_dir, "rw_string_10.mv"))
    data_matrix = DataMatrixIO.loadOffHeapDataMatrix(rw_path)

    # PhenIX data directory
    phenix_dir = JPaths.get(os.path.join(phenotype_data_dir, "phenix"))

    factory = PriorityFactoryImpl(priority_service, data_matrix, phenix_dir)
    return factory, ds


METRIC_KEYS = ["mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@100", "auc"]
TEX_HEADER = "MR & MRR & Hits@1 & Hits@3 & Hits@10 & Hits@100 & AUC"


def parse_folds(folds_str):
    if folds_str.strip().lower() == "all":
        return list(range(10))
    return [int(f) for f in folds_str.split(",") if f.strip()]


def tex_row(metrics):
    return " & ".join(f"{metrics[k]:.3f}" for k in METRIC_KEYS)


def mean_std_row(all_metrics):
    parts = []
    for k in METRIC_KEYS:
        vals = np.array([m[k] for m in all_metrics], dtype=float)
        parts.append(f"{vals.mean():.3f} ± {vals.std():.3f}")
    return " & ".join(parts)


def emit(summary_f, text):
    """Write to both stdout and the summary file."""
    print(text)
    if summary_f is not None:
        summary_f.write(text + "\n")
        summary_f.flush()


def run_fold(fold, prioritisers, disease2hpo, eval_genes, gene_to_index,
             java_genes, entrez_to_eval_idx, summary_f):
    """Run all prioritisers on one fold. Returns {pname: macro_metrics}."""
    test_disease_genes = pd.read_csv(
        f"data/folds/fold_{fold}/test.csv", sep="\t")
    test_pairs = [(row["Disease"], row["Gene"])
                  for _, row in test_disease_genes.iterrows()]
    logger.info(f"Fold {fold}: {len(test_pairs)} test pairs, "
                f"{len(eval_genes)} eval genes")

    fold_metrics = {}
    for pname, prioritiser in prioritisers.items():
        logger.info(f"Running {pname} prioritiser on fold {fold}...")
        results_data = []

        for test_disease, test_gene in tqdm(test_pairs,
                                            desc=f"{pname} fold {fold}"):
            hpo_ids = disease2hpo.get(test_disease, [])
            if not hpo_ids:
                scores = [0.0] * len(eval_genes)
                results_data.append((test_gene, test_disease,
                                     gene_to_index[test_gene], scores))
                continue

            java_hpo = ArrayList()
            for h in hpo_ids:
                java_hpo.add(h)

            result_list = (prioritiser
                           .prioritise(java_hpo, java_genes)
                           .collect(Collectors.toList()))

            scores = [0.0] * len(eval_genes)
            for r in result_list:
                gid = int(r.getGeneId())
                if gid in entrez_to_eval_idx:
                    scores[entrez_to_eval_idx[gid]] = float(r.getScore())

            results_data.append((test_gene, test_disease,
                                 gene_to_index[test_gene], scores))

        out_file = f"data/results/exomiser_{pname}_fold_{fold}.tsv"
        with open(out_file, "w") as f:
            for gene, disease, gene_index, scores in results_data:
                f.write(f"{gene}\t{disease}\t{gene_index}\t"
                        + "\t".join(str(s) for s in scores) + "\n")

        _, macro_metrics = compute_metrics(out_file, verbose=False)
        emit(summary_f, f"Exomiser {pname} (fold {fold})")
        emit(summary_f, TEX_HEADER)
        emit(summary_f, tex_row(macro_metrics))
        emit(summary_f, "")
        fold_metrics[pname] = macro_metrics

    return fold_metrics


@ck.command()
@ck.option("--folds", type=str, default="all",
           help="Fold number, comma-separated list (e.g. '0,1,2'), or 'all'")
def main(folds):
    fold_list = parse_folds(folds)

    if not os.path.exists("data/results"):
        os.makedirs("data/results")

    logger.info("Initializing Exomiser prioritisers...")
    factory, ds = build_priority_factory()
    logger.info("Exomiser ready.")

    # Shared data (independent of fold)
    disease_phenotypes = pd.read_csv("data/disease_phenotypes.csv")
    all_gene_diseases = pd.read_csv("data/gene_diseases.csv")

    eval_genes = sorted(list(set(all_gene_diseases["Gene"].values)))
    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}

    # disease -> list of HPO IDs (HP:XXXXXXX format)
    disease2hpo = {}
    for _, row in disease_phenotypes.iterrows():
        disease = row["Disease"]
        hpo_id = uri_to_hpo(row["Phenotype"])
        if hpo_id.startswith("HP:"):
            disease2hpo.setdefault(disease, []).append(hpo_id)

    # Java Gene objects for all eval genes (built once, reused across folds)
    java_genes = ArrayList()
    entrez_to_eval_idx = {}
    for i, gene_uri in enumerate(eval_genes):
        entrez_id = uri_to_entrez(gene_uri)
        java_genes.add(Gene(str(entrez_id), entrez_id))
        entrez_to_eval_idx[entrez_id] = i

    # Prioritisers built once — reused across folds
    prioritisers = {
        "hiphive": factory.makeHiPhivePrioritiser(HiPhiveOptions.defaults()),
        "phive": factory.makePhivePrioritiser(),
        "phenix": factory.makePhenixPrioritiser(),
    }

    # per_fold[pname] = list of macro_metrics dicts, one per fold
    per_fold = {pname: [] for pname in prioritisers}

    folds_tag = "all" if fold_list == list(range(10)) else \
        "_".join(str(f) for f in fold_list)
    summary_path = f"data/results/exomiser_metrics_summary_folds_{folds_tag}.txt"
    logger.info(f"Writing metrics summary to {summary_path}")

    with open(summary_path, "w") as summary_f:
        emit(summary_f, f"# Exomiser evaluation — folds: {fold_list}")
        emit(summary_f, "")

        for fold in fold_list:
            fold_metrics = run_fold(fold, prioritisers, disease2hpo,
                                    eval_genes, gene_to_index, java_genes,
                                    entrez_to_eval_idx, summary_f)
            for pname, m in fold_metrics.items():
                per_fold[pname].append(m)

        if len(fold_list) > 1:
            emit(summary_f, "=" * 60)
            emit(summary_f,
                 f"Aggregated across {len(fold_list)} folds: {fold_list}")
            emit(summary_f, "=" * 60)
            emit(summary_f, "")
            for pname, metrics_list in per_fold.items():
                emit(summary_f,
                     f"Exomiser {pname} (mean ± std, "
                     f"{len(metrics_list)} folds)")
                emit(summary_f, TEX_HEADER)
                emit(summary_f, mean_std_row(metrics_list))
                emit(summary_f, "")

    ds.close()


if __name__ == "__main__":
    main()
