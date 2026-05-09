import os
import glob
import jpype
import importlib.util

# Get mowl jar path without importing mowl (JVM not started yet)
mowl_spec = importlib.util.find_spec("mowl")
mowl_path = os.path.dirname(mowl_spec.origin)
mowl_jars_dir = os.path.join(mowl_path, "lib")
mowl_jars = glob.glob(os.path.join(mowl_jars_dir, "*.jar"))

if not mowl_jars:
    raise FileNotFoundError(f"Could not find mOWL jars in {mowl_jars_dir}")

# my_custom_jars = ["/home/zhapacfp/Git/multihop-gda/build/OWL2VecStarGDAProjector.jar"]
full_classpath = mowl_jars # + my_custom_jars

jpype.startJVM(
    jpype.getDefaultJVMPath(),
    "-ea",
    "-Xmx10g",
    classpath=full_classpath,
    convertStrings=False
)

import mowl
mowl.init_jvm("4g")

from mowl.projection import OWL2VecStarProjector, Edge
from mowl.datasets import PathDataset
from mowl.utils.random import seed_everything
from pykeen.models import TransD
from pykeen.training import SLCWATrainingLoop
from pykeen.training.callbacks import StopperTrainingCallback
import torch as th
from torch.optim import Adam
import click as ck
import pandas as pd
import wandb
import tomllib
from tqdm import tqdm

from data import create_train_val_split
from pykeen_utils import ValidationStopper
from evaluation import evaluate_by_similarity, evaluate_by_graph

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def model_resolver(triples_factory, embedding_dim, random_seed):
    model = TransD(
        triples_factory=triples_factory, 
        embedding_dim=embedding_dim,
        relation_dim=embedding_dim,
        random_seed=random_seed
    )
    return model

                 
@ck.command()
@ck.option("--fold", type=int, default=0, help="Fold number for the dataset")
@ck.option("--use_phenotypes", '-pheno', is_flag=True, help="Use gene phenotype information")
@ck.option("--use_functions", '-func', is_flag=True, help="Use gene function information")
@ck.option("--use_site", '-site', is_flag=True, help="Use gene site information")
@ck.option("--projector_name", type=ck.Choice(["owl2vecstar", "owl2vecstar_gda"]), default="owl2vecstar", help="Projector to use for ontology projection")
@ck.option("--embedding_dim", type=int, default=400, help="Embedding dimension for entities")
@ck.option("--batch_size", type=int, default=8192, help="Batch size for training")
@ck.option("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
@ck.option("--random_seed", type=int, default=0, help="Random seed for reproducibility")
@ck.option("--only_test", "-ot", is_flag=True, help="Only test the model")
@ck.option("--use_graph", "-graph", is_flag=True, help="Use 2P evaluation (indirectly_causes) instead of standard evaluation")
@ck.option("--description", type=str, default="", help="Description for the wandb run")
@ck.option("--no_sweep", is_flag=True, help="Disable wandb sweep mode")
def main(fold, use_phenotypes, use_functions, use_site,
         projector_name, embedding_dim, batch_size,
         learning_rate, random_seed, only_test, use_graph, description,
         no_sweep):


    if not os.path.exists("data/results"):
        os.makedirs("data/results")
    if not os.path.exists("data/models"):
        os.makedirs("data/models")
    
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    wandb.init(entity=config["wandb"]["entity"], project=config["wandb"]["project"], name=description)
    if no_sweep:
        wandb.log({"embedding_dim": embedding_dim,
                   "batch_size": batch_size,
                   "learning_rate": learning_rate,
                   "fold": fold,
                   "use_phenotypes": use_phenotypes,
                   "use_functions": use_functions,
                   "use_site": use_site,
                   })
    else:
        embedding_dim = wandb.config.embedding_dim
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        fold = wandb.config.fold
        
    seed_everything(random_seed)
    
    train_gene_diseases = pd.read_csv(f"data/folds/fold_{fold}/train.csv", sep="\t")
    # Split into train and validation ensuring all validation entities are in training
    train_disease_genes, val_disease_genes = create_train_val_split(train_gene_diseases, val_ratio=0.1, random_seed=random_seed)

    train_diseases = sorted(list(set(train_disease_genes['Disease'].values)))
    val_diseases = sorted(list(set(val_disease_genes['Disease'].values)))

    non_test_diseases = set(train_diseases) | set(val_diseases)

    test_disease_genes = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
    test_diseases = set(test_disease_genes['Disease'].values)

    overlap = test_diseases & non_test_diseases
    assert not overlap, (
        f"Test diseases overlap with train/val diseases ({len(overlap)} diseases): {overlap}"
    )
    logger.info(f"Split check passed: {len(test_diseases)} test diseases, "
                f"{len(non_test_diseases)} train/val diseases, 0 overlap.")

    upheno_edges_file = "data/upheno_edges_gda.tsv" if projector_name == "owl2vecstar_gda" else "data/upheno_edges.tsv"
    go_edges_file = "data/go_edges.tsv"
    uberon_edges_file = "data/uberon_edges.tsv"
    projector = OWL2VecStarProjector(bidirectional_taxonomy=True)

    if not os.path.exists(go_edges_file) and use_functions:
        ds = PathDataset("data/go.owl")
        train_edges = projector.project(ds.ontology)
        with open(go_edges_file, "w") as f:
            for edge in train_edges:
                f.write(f"{edge.src}\t{edge.rel}\t{edge.dst}\n")

    if not os.path.exists(uberon_edges_file) and use_site:
        ds = PathDataset("data/uberon.owl")
        train_edges = projector.project(ds.ontology)
        with open(uberon_edges_file, "w") as f:
            for edge in train_edges:
                f.write(f"{edge.src}\t{edge.rel}\t{edge.dst}\n")
                
    triples = []
    entities = set()
    phenos = set()
    relations = set()
    with open(upheno_edges_file, "r") as f:
        for line in f:
            src, rel, dst = line.strip().split("\t")
            triples.append((src, rel, dst))
            entities.add(src)
            entities.add(dst)
            relations.add(rel)

    if use_functions:
        with open(go_edges_file, "r") as f:
            for line in f:
                src, rel, dst = line.strip().split("\t")
                triples.append((src, rel, dst))
                entities.add(src)
                entities.add(dst)
                relations.add(rel)

    if use_site:
        with open(uberon_edges_file, "r") as f:
            for line in f:
                src, rel, dst = line.strip().split("\t")
                triples.append((src, rel, dst))
                entities.add(src)
                entities.add(dst)
                relations.add(rel)
            
    disease_phenotypes = pd.read_csv("data/disease_phenotypes.csv")  # Always needed for evaluation

    completed_annots = 0
    missing_annots = 0
    for row in disease_phenotypes.itertuples(index=False):
        disease = row.Disease
        phenotype = row.Phenotype
        if phenotype not in entities:
            missing_annots += 1
        else:
            if disease in test_diseases:
                continue
            completed_annots += 1
            triples.append((disease, 'has_symptom', phenotype))
            entities.add(disease)
    logger.info(f"Completed disease-phenotype annotations: {completed_annots}, Missing annotations: {missing_annots}")

    if use_functions:
        gene_functions = pd.read_csv("data/gene_functions.csv")
        ignored_functions = 0
        for row in tqdm(gene_functions.itertuples(index=False), leave=False, total=len(gene_functions), desc="Adding gene-function associations"):
            gene = row.Gene
            function = row.Function
            if function not in entities:
                ignored_functions += 1
                continue
            triples.append((gene, 'has_function', function))
            entities.add(gene)
            entities.add(function)
        logger.info(f"Gene-function associations added: {len(gene_functions) - ignored_functions}, Ignored (function not in graph): {ignored_functions}")

    if use_site:
        gene_sites = pd.read_csv("data/gene_site.csv")
        ignored_sites = 0
        for row in tqdm(gene_sites.itertuples(index=False), leave=False, total=len(gene_sites), desc="Adding gene-site associations"):
            gene = row.Gene
            site = row.Tissue
            if site not in entities:
                ignored_sites += 1
                continue
            triples.append((gene, 'expressed_in', site))
            entities.add(gene)
            entities.add(site)
        logger.info(f"Gene-site associations added: {len(gene_sites) - ignored_sites}, Ignored (site not in graph): {ignored_sites}")

    # Gene-phenotype: always load; add all triples if use_phenotypes,
    # else only add triples for genes not yet in entities (prevents gene loss)
    gene_phenotypes = pd.read_csv("data/gene_phenotypes.csv")
    completed_annots = 0
    missing_annots = 0
    genes_with_pheno = set()
    for row in gene_phenotypes.itertuples(index=False):
        gene = row.Gene
        phenotype = row.Phenotype
        if phenotype not in entities:
            missing_annots += 1
        elif use_phenotypes or gene not in entities:
            completed_annots += 1
            triples.append((gene, 'has_phenotype', phenotype))
            entities.add(gene)
            genes_with_pheno.add(gene)
    logger.info(f"Completed gene-phenotype annotations: {completed_annots}, Missing annotations: {missing_annots}")
    all_pheno_genes = set(gene_phenotypes['Gene'].values)
    genes_with_0_phenos = all_pheno_genes - genes_with_pheno
    logger.info(f"Genes with 0 phenotypes in graph: {len(genes_with_0_phenos)} / {len(all_pheno_genes)}")

    assert len(test_diseases & non_test_diseases) == 0, "Test diseases overlap with train diseases"
    assert len(test_diseases & entities) == 0, "Test diseases overlap with graph diseases"

    for row in tqdm(train_disease_genes.itertuples(index=False), leave=False, total=len(train_disease_genes), desc="Adding gene-disease associations"):
        disease = row.Disease
        gene = row.Gene
        triples.append((gene, 'associated_with', disease))
        if use_phenotypes:
            assert gene in entities, f"Gene {gene} not in entities"
        assert disease in entities, f"Disease {disease} not in entities"

    entities = sorted(list(entities))
    relations = sorted(list(relations))
    entities_set = set(entities)

    # Build gene2pheno, gene2function, gene2site mappings (needed for indirect triples and evaluation)
    gene2pheno = dict()
    if use_phenotypes:
        used_phenos = 0
        ignored_phenos = 0
        for row in tqdm(gene_phenotypes.itertuples(index=False), leave=False, total=len(gene_phenotypes), desc="Building gene2pheno mapping"):
            gene = row.Gene
            phenotype = row.Phenotype
            if phenotype not in entities_set:
                ignored_phenos += 1
            else:
                used_phenos += 1
                if gene not in gene2pheno:
                    gene2pheno[gene] = []
                gene2pheno[gene].append(phenotype)
        logger.info(f"Gene-Phenotype associations used: {used_phenos}, ignored (phenotype not in graph): {ignored_phenos}. Total genes with phenotypes: {len(gene2pheno)}")

    gene2function = dict()
    if use_functions:
        ignored_functions = 0
        for row in gene_functions.itertuples(index=False):
            gene = row.Gene
            function = row.Function
            if function not in entities_set:
                ignored_functions += 1
                continue
            if gene not in gene2function:
                gene2function[gene] = []
            gene2function[gene].append(function)
        logger.info(f"Gene-function associations used for validation/testing: {len(gene_functions) - ignored_functions}, ignored (function not in graph): {ignored_functions}. Total genes with functions: {len(gene2function)}")

    gene2site = dict()
    if use_site:
        for row in gene_sites.itertuples(index=False):
            gene = row.Gene
            site = row.Tissue
            if site not in entities_set:
                continue
            if gene not in gene2site:
                gene2site[gene] = []
            gene2site[gene].append(site)

    disease2pheno = dict()
    used_phenos = 0
    ignored_phenos = 0
    for row in disease_phenotypes.itertuples(index=False):
        disease = row.Disease
        phenotype = row.Phenotype
        if phenotype not in entities_set:
            ignored_phenos += 1
        else:
            used_phenos += 1
            if disease not in disease2pheno:
                disease2pheno[disease] = []
            disease2pheno[disease].append(phenotype)
    logger.info(f"Disease-Phenotype associations used: {used_phenos}, ignored (phenotype not in graph): {ignored_phenos}. Total diseases with phenotypes: {len(disease2pheno)}")

    # Add (gene, causes_phenotype, symptom) triples via gene -> disease -> symptom
    causes_pheno_count = 0
    for row in tqdm(train_disease_genes.itertuples(index=False), leave=False, total=len(train_disease_genes), desc="Adding causes_phenotype triples"):
        gene = row.Gene
        disease = row.Disease
        for symptom in disease2pheno.get(disease, []):
            triples.append((gene, 'causes_phenotype', symptom))
            causes_pheno_count += 1
    logger.info(f"Gene causes_phenotype triples added: {causes_pheno_count}")

    triples = sorted(triples)

    triple_entities = set(e for src, _, dst in triples for e in (src, dst))
    leaked = test_diseases & triple_entities
    assert not leaked, (
        f"{len(leaked)} test disease(s) found in triples: {leaked}"
    )
    logger.info("Leakage check passed: no test diseases found in triples.")

    mowl_triples = [Edge(src, rel, dst) for src, rel, dst in triples]
    triples_factory = Edge.as_pykeen(mowl_triples)

    model = model_resolver(triples_factory, embedding_dim, random_seed).to("cuda")

    sources = []
    if use_phenotypes:
        sources.append("pheno")
    if use_functions:
        sources.append("func")
    if use_site:
        sources.append("expr")
        
    source_str = "_".join(sources) if sources else "base"

    file_identifier = f"transd_fold_{fold}_seed_{random_seed}_dim_{embedding_dim}_bs_{batch_size}_lr_{learning_rate}_{source_str}_proj_{projector_name}_use_graph_{use_graph}"
    model_out_filename = f"data/models/{file_identifier}.pt"

    all_gene_diseases = pd.read_csv("data/gene_diseases.csv")
    eval_genes = set(all_gene_diseases['Gene'].values)
    logger.info(f"Number of evaluation genes: {len(eval_genes)}")
    eval_genes = sorted(list(eval_genes))

    tolerance = 5
    validation_stopper = ValidationStopper(
        model,
        triples_factory,
        file_identifier,
        val_disease_genes,
        gene2pheno,
        disease2pheno,
        eval_genes,
        tolerance,
        model_out_filename,
        use_graph=use_graph
    )

    validation_callback = StopperTrainingCallback(stopper=validation_stopper, triples_factory=triples_factory, best_epoch_model_file_path=model_out_filename)

    optimizer = Adam(params=model.get_grad_params(), lr=learning_rate)

    if not only_test:

        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=triples_factory,
            optimizer=optimizer
        )

        _ = training_loop.train(
            triples_factory=triples_factory,
            num_epochs=1000,
            batch_size=batch_size,
            callbacks=[validation_callback],
        )

    print("Training complete. Loading best model for testing...")


    model.load_state_dict(th.load(model_out_filename, weights_only=True))

    # Evaluate on test set
    output_prefix = f"data/results/kge_results_{file_identifier}"

    if use_graph:
        (inductive_bma_macro_metrics,
         inductive_bmm_macro_metrics) = evaluate_by_graph(
             model=model,
             test_disease_genes=test_disease_genes,
             disease2pheno=disease2pheno,
             eval_genes=eval_genes,
             triples_factory=triples_factory,
             output_file_prefix=output_prefix,
             verbose=True
        )
    else:
        (inductive_bma_macro_metrics,
         inductive_bmm_macro_metrics) = evaluate_by_similarity(
             model=model,
             test_disease_genes=test_disease_genes,
             gene2pheno=gene2pheno,
             disease2pheno=disease2pheno,
             eval_genes=eval_genes,
             triples_factory=triples_factory,
             output_file_prefix=output_prefix,
             verbose=True
        )

    # Log test metrics to wandb
    metrics = ['mr', 'mrr', 'auc', 'hits@1', 'hits@3', 'hits@10', 'hits@100']
    bma_macro_to_log = {f"test_imac_bma_{k}": v for k, v in inductive_bma_macro_metrics.items() if k in metrics}
    bmm_macro_to_log = {f"test_imac_bmm_{k}": v for k, v in inductive_bmm_macro_metrics.items() if k in metrics}
    wandb.log(bma_macro_to_log)
    wandb.log(bmm_macro_to_log)

    
if __name__ == "__main__":
    main()
