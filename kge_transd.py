import mowl
mowl.init_jvm("10g")

from mowl.projection import OWL2VecStarProjector, Edge
from mowl.datasets import PathDataset
from mowl.utils.random import seed_everything
from pykeen.models import TransD
from pykeen.training import SLCWATrainingLoop
from pykeen.training.callbacks import StopperTrainingCallback
import torch as th
from torch.optim import Adam
import os
import click as ck
import pandas as pd
import wandb
import tomllib
from tqdm import tqdm

from data import create_train_val_split
from pykeen_utils import ValidationStopper
from evaluation import evaluate_model

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
@ck.option("--graph2", is_flag=True, help="Use graph2")
@ck.option("--graph3", is_flag=True, help="Use graph3")
@ck.option("--graph4", is_flag=True, help="Use graph4")
@ck.option("--projector_name", type=ck.Choice(["owl2vecstar"]), default="owl2vecstar", help="Projector to use for ontology projection")
@ck.option("--embedding_dim", type=int, default=100, help="Embedding dimension for entities")
@ck.option("--batch_size", type=int, default=2048, help="Batch size for training")
@ck.option("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
@ck.option("--random_seed", type=int, default=0, help="Random seed for reproducibility")
@ck.option("--only_test", "-ot", is_flag=True, help="Only test the model")
@ck.option("--description", type=str, default="", help="Description for the wandb run")
@ck.option("--no_sweep", is_flag=True, help="Disable wandb sweep mode")
def main(fold, graph2, graph3, graph4, projector_name, embedding_dim,
         batch_size, learning_rate, random_seed, only_test,
         description, no_sweep):


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
                   })
    else:
        embedding_dim = wandb.config.embedding_dim
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        fold = wandb.config.fold
        
    seed_everything(random_seed)
    
    if graph4:
        graph3 = True
    if graph3:
        graph2 = True

    train_gene_diseases = pd.read_csv(f"data/folds/fold_{fold}/train.csv", sep="\t")
    # Split into train and validation ensuring all validation entities are in training
    train_disease_genes, val_disease_genes = create_train_val_split(train_gene_diseases, val_ratio=0.1, random_seed=random_seed)

    train_diseases = sorted(list(set(train_disease_genes['Disease'].values)))
    val_diseases = sorted(list(set(val_disease_genes['Disease'].values)))

    non_test_diseases = set(train_diseases) | set(val_diseases)

    test_disease_genes = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
    test_diseases = set(test_disease_genes['Disease'].values)

    upheno_edges_file = "data/upheno_edges.tsv"
    go_edges_file = "data/go_edges.tsv"
    projector = OWL2VecStarProjector(bidirectional_taxonomy=True)
    
    if not os.path.exists(upheno_edges_file):
        ds = PathDataset("data/upheno.owl")
        train_edges = projector.project(ds.ontology)
        with open(upheno_edges_file, "w") as f:
            for edge in train_edges:
                f.write(f"{edge.src}\t{edge.rel}\t{edge.dst}\n")

    if not os.path.exists(go_edges_file):
        ds = PathDataset("data/go.owl")
        train_edges = projector.project(ds.ontology)
        with open(go_edges_file, "w") as f:
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
    with open(go_edges_file, "r") as f:
        for line in f:
            src, rel, dst = line.strip().split("\t")
            triples.append((src, rel, dst))
            entities.add(src)
            entities.add(dst)
            relations.add(rel)
            
    gene_phenotypes = pd.read_csv("data/gene_phenotypes.csv")
    disease_phenotypes = pd.read_csv("data/disease_phenotypes.csv")
    gene_functions = pd.read_csv("data/gene_functions.csv")
    
    if graph2:
        completed_annots = 0
        missing_annots = 0
        for _, row in gene_phenotypes.iterrows():
            gene = row['Gene']
            phenotype = row['Phenotype']

            if phenotype not in entities:
                missing_annots += 1
            else:
                completed_annots += 1
                triples.append((gene, 'has_phenotype', phenotype))
                entities.add(gene)
        logger.info(f"Graph2 - Completed gene-phenotype annotations: {completed_annots}, Missing annotations: {missing_annots}")
                
    if graph3:
        completed_annots = 0
        missing_annots = 0
        for _, row in disease_phenotypes.iterrows():
            disease = row['Disease']
            phenotype = row['Phenotype']
            if phenotype not in entities:
                missing_annots += 1
            else:
                if disease in test_diseases:
                    continue
                completed_annots += 1
                triples.append((disease, 'has_symptom', phenotype))
                entities.add(disease)
        logger.info(f"Graph3 - Completed disease-phenotype annotations: {completed_annots}, Missing annotations: {missing_annots}")
                
    for _, row in tqdm(gene_functions.iterrows(), leave=False, total=len(gene_functions), desc="Adding gene-function associations"):
        gene = row['Gene']
        function = row['Function']
        assert function in entities, f"Function {function} not in entities"
        triples.append((gene, 'has_function', function))
        entities.add(gene)
        entities.add(function)
            
    assert len(test_diseases & non_test_diseases) == 0, "Test diseases overlap with train diseases"

    assert len(test_diseases & entities) == 0, "Test diseases overlap with graph diseases"
                        
    if graph4:
        for _, row in tqdm(train_disease_genes.iterrows(), leave=False, total=len(train_disease_genes), desc="Adding gene-disease associations for graph4"):
            disease = row['Disease']
            gene = row['Gene']
            triples.append((gene, 'associated_with', disease))
            assert gene in entities, f"Gene {gene} not in entities"
            assert disease in entities, f"Disease {disease} not in entities"
            
    entities = sorted(list(entities))
    relations = sorted(list(relations))

    triples = sorted(triples)
    mowl_triples = [Edge(src, rel, dst) for src, rel, dst in triples]
    triples_factory = Edge.as_pykeen(mowl_triples)
    
    model = model_resolver(triples_factory, embedding_dim, random_seed).to("cuda")

    graph_status = "graph4" if graph4 else "graph3" if graph3 else "graph2" if graph2 else "graph1"

    file_identifier = f"transd_fold_{fold}_seed_{random_seed}_dim_{embedding_dim}_bs_{batch_size}_lr_{learning_rate}_{graph_status}"
    model_out_filename = f"data/models/{file_identifier}.pt"

    # Build gene2pheno and gene2function disease2pheno mappings (needed for validation and testing)
    gene2pheno = dict()
    used_phenos = 0
    ignored_phenos = 0
    entities_set = set(entities)
    for _, row in tqdm(gene_phenotypes.iterrows(), leave=False, total=len(gene_phenotypes), desc="Building gene2pheno mapping"):
        gene = row['Gene']
        phenotype = row['Phenotype']
        if phenotype not in entities_set:
            ignored_phenos += 1
        else:
            used_phenos += 1
            if gene not in gene2pheno:
                gene2pheno[gene] = []
            gene2pheno[gene].append(phenotype)
    logger.info(f"Gene-Phenotype associations used: {used_phenos}, ignored (phenotype not in graph): {ignored_phenos}. Total genes with phenotypes: {len(gene2pheno)}")
            
    disease2pheno = dict()
    used_phenos = 0
    ignored_phenos = 0
    for _, row in disease_phenotypes.iterrows():
        disease = row['Disease']
        phenotype = row['Phenotype']
        if phenotype not in entities_set:
            ignored_phenos += 1
        else:
            used_phenos += 1
            if disease not in disease2pheno:
                disease2pheno[disease] = []
            disease2pheno[disease].append(phenotype)
    logger.info(f"Disease-Phenotype associations used: {used_phenos}, ignored (phenotype not in graph): {ignored_phenos}. Total diseases with phenotypes: {len(disease2pheno)}")
            
    gene2function = dict()
    for _, row in gene_functions.iterrows():
        gene = row['Gene']
        function = row['Function']
        if gene not in gene2function:
            gene2function[gene] = []
        gene2function[gene].append(function)
        
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
        gene2function,
        disease2pheno,
        eval_genes,
        graph3,
        graph4,
        tolerance,
        model_out_filename
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

    (inductive_bma_macro_metrics,
     inductive_bmm_macro_metrics) = evaluate_model(
         model=model,
         test_disease_genes=test_disease_genes,
         gene2pheno=gene2pheno,
         disease2pheno=disease2pheno,
         eval_genes=eval_genes,
         triples_factory=triples_factory,
         graph3=graph3,
         graph4=graph4,
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
