import click as ck
import wandb
import tomllib
import os
import json
import numpy as np
import pandas as pd
import logging
from util import seed_everything
from data import create_train_val_split
import torch as th
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from geometre.model import GeometrE
from geometre.dataloader import TrainDataset, DisjointDataset, SingledirectionalOneShotIterator
from evaluation import evaluate_qa_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)



query_pattern_to_name = {
    "Anchor": "sub",
    "I(Anchor,P(Anchor))": "pi",
    "I(Anchor,P(Anchor),P(Anchor))": "ppi",
    "P(Anchor)": "1p",
    "P(P(Anchor))": "2p",
    "P(I(Anchor,P(Anchor),P(Anchor)))": "ppip",
    "P(I(Anchor,P(Anchor)))": "pip",
    "P(I(Anchor,P(I(Anchor,P(Anchor))),P(Anchor)))": "humanoid",
}

def load_mapping(filename):
    mapping = {}
    with open(filename, "r") as f:
        for line in f:
            key, value = line.strip().split("\t")
            mapping[key] = int(value)

    return mapping


@ck.command()
@ck.option("--fold", type=int, default=0, help="Fold number for the dataset")
@ck.option("--use_phenotypes", '-pheno', is_flag=True, help="Use gene phenotype information")
@ck.option("--use_functions", '-func', is_flag=True, help="Use gene function information")
@ck.option("--use_site", '-site', is_flag=True, help="Use gene site of expression information")
@ck.option("--embedding_dim", type=int, default=200, help="Embedding dimension for entities")
# @ck.option("--batch_size", type=int, default=32768, help="Batch size for training")
@ck.option("--batch_size", type=int, default=8192, help="Batch size for training")
@ck.option("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")
@ck.option("--gamma", type=float, default=10, help="Margin for the loss function")
@ck.option("--alpha", type=float, default=0.2, help="Weight for the box in-distance function")
@ck.option("--with_answer_embedding", is_flag=True, help="Whether to include answer embedding ")
@ck.option("--negative_sample_size", type=int, default=128, help="Number of negative samples per positive sample")
@ck.option("--min_queries_per_pattern", '-min', type=int, default=1000, help="Minimum number of queries per pattern to be included in training")
@ck.option("--random_seed", type=int, default=0, help="Random seed for reproducibility")
@ck.option("--max_epochs", type=int, default=100000, help="Number of training epochs")
@ck.option("--validate_every", type=int, default=50, help="Number of epochs between validations")
@ck.option("--do_train", "-train", is_flag=True, help="Only test the model")
@ck.option("--do_valid", "-valid", is_flag=True, help="Only validate the model")
@ck.option("--do_test", "-test", is_flag=True, help="Only test the model")
@ck.option("--description", type=str, default="", help="Description for the wandb run")
@ck.option("--no_sweep", is_flag=True, help="Disable wandb sweep mode")
@ck.option("--device", type=str, default="cuda", help="Device to use for training (cuda/cpu)")
@ck.option("--dispersion_weight", type=float, default=0.5, help="Weight for gene embedding dispersion loss (0 = disabled)")
def main(fold, use_phenotypes, use_functions, use_site, embedding_dim,
         batch_size, learning_rate, gamma, alpha,
         with_answer_embedding, negative_sample_size,
         min_queries_per_pattern, random_seed, max_epochs,
         validate_every, do_train, do_valid, do_test, description,
         no_sweep, device, dispersion_weight):

    seed_everything(random_seed)
    
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    wandb_logger = wandb.init(entity=config["wandb"]["entity"], project=config["wandb"]["project"], name=description)
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
        
    device = device or ("cuda" if th.cuda.is_available() else "cpu")
    
    
    entity_to_id = load_mapping("data/entity_to_id.txt")
    relation_to_id = load_mapping("data/relation_to_id.txt")

    pattern_to_queries = dict()
    with open("data/pattern_to_queries.txt", "r") as f:
        for line in f:
            pattern, query = line.strip().split("\t")
            if pattern not in pattern_to_queries:
                pattern_to_queries[pattern] = set()
            pattern_to_queries[pattern].add(query)

    pattern_to_queries = {pattern: queries for pattern, queries in pattern_to_queries.items() if len(queries) >= min_queries_per_pattern}
    for pattern, queries in pattern_to_queries.items():
        print(f"Found {len(queries)} queries for pattern {pattern}")

    query_to_pattern = list()
    for pattern, queries in pattern_to_queries.items():
        for query in queries:
            query_to_pattern.append((query, pattern))
            
    query_to_answers = dict()
    with open("data/query_to_answers.txt", "r") as f:
        for line in f:
            query, answers = line.strip().split("\t")
            if query not in query_to_answers:
                query_to_answers[query] = set()
            answers = answers.split(" ")
            answers = [int(answer) for answer in answers]
            query_to_answers[query].update(answers)

    train_gene_diseases = pd.read_csv(f"data/folds/fold_{fold}/train.csv", sep="\t")
    # Split into train and validation ensuring all validation entities are in training
    train_disease_genes, val_disease_genes = create_train_val_split(train_gene_diseases, val_ratio=0.1, random_seed=random_seed)

    train_diseases = sorted(list(set(train_disease_genes['Disease'].values)))
    val_diseases = sorted(list(set(val_disease_genes['Disease'].values)))

    for disease in train_diseases:
        entity_to_id[disease] = len(entity_to_id)

    non_test_diseases = set(train_diseases) | set(val_diseases)

    test_disease_genes = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
    test_diseases = set(test_disease_genes['Disease'].values)

    disease_phenotypes = pd.read_csv("data/disease_phenotypes.csv")  # Always needed for evaluation
    for disease in disease_phenotypes['Disease'].values:
        if disease  in val_diseases or disease in test_diseases:
            continue
        if not disease in entity_to_id:
            entity_to_id[disease] = len(entity_to_id)


    
    triples = []
    entities = set(entity_to_id.keys())
    

    if use_phenotypes:
        gene_phenotypes = pd.read_csv("data/gene_phenotypes.csv")
        # Individual 1p gene-phenotype queries removed: gene phenotypes are now
        # handled as a group via kip intersection queries. Relation ID still registered.
        # completed_annots = 0
        # missing_annots = 0
        # for _, row in gene_phenotypes.iterrows():
        #     gene = row['Gene']
        #     phenotype = row['Phenotype']
        #     if phenotype not in entities:
        #         missing_annots += 1
        #     else:
        #         completed_annots += 1
        #         query_pattern = "P(Anchor)"
        #         assert gene in entity_to_id, f"Gene {gene} not in entity_to_id"
        #         assert phenotype in entity_to_id, f"Phenotype {phenotype} not in entity_to_id"
        #         relation = "has_phenotype"
        #         if relation not in relation_to_id:
        #             relation_to_id[relation] = len(relation_to_id)
        #         query = f"P({relation_to_id[relation]},{entity_to_id[gene]})"
        #         answer = entity_to_id[phenotype]
        #         query_to_pattern.append((query, query_pattern))
        #         if query not in query_to_answers:
        #             query_to_answers[query] = set()
        #         query_to_answers[query].add(answer)
        # logger.info(f"Completed gene-phenotype annotations: {completed_annots}, Missing annotations: {missing_annots}")
        if "has_phenotype" not in relation_to_id:
            relation_to_id["has_phenotype"] = len(relation_to_id)
        logger.info(f"Loaded {len(gene_phenotypes)} gene-phenotype annotations")

    completed_annots = 0
    missing_annots = 0
    for _, row in disease_phenotypes.iterrows():
        disease = row['Disease']
        phenotype = row['Phenotype']
        if phenotype not in entities:
            missing_annots += 1
        else:
            if disease in val_diseases or disease in test_diseases:
                continue
            completed_annots += 1

            assert disease in entity_to_id, f"Disease {disease} not in entity_to_id. Disease in training {disease in train_diseases}, Disease in valid {disease in val_diseases}. Disease in test {disease in test_diseases}"
            assert phenotype in entity_to_id, f"Phenotype {phenotype} not in entity_to_id"
            # query_pattern = "P(Anchor)"

            # relation = "has_symptom"
            # if relation not in relation_to_id:
                # relation_to_id[relation] = len(relation_to_id)

            # query = f"P({relation_to_id[relation]},{entity_to_id[phenotype]})"
            # answer = entity_to_id[disease]

            # query_to_pattern.append((query, query_pattern))
            # if query not in query_to_answers:
                # query_to_answers[query] = set()
            # query_to_answers[query].add(answer)


    logger.info(f"Completed disease-phenotype annotations: {completed_annots}, Missing annotations: {missing_annots}")
    if "has_phenotype" not in relation_to_id:
        relation_to_id["has_phenotype"] = len(relation_to_id)
    if "has_symptom" not in relation_to_id:
        relation_to_id["has_symptom"] = len(relation_to_id)

    if use_functions:
        gene_functions = pd.read_csv("data/gene_functions.csv")
        # Individual 1p gene-function queries removed: gene functions are now
        # handled as a group via kip intersection queries. Relation ID still registered.
        # ignored_functions = 0
        # for _, row in tqdm(gene_functions.iterrows(), leave=False, total=len(gene_functions), desc="Adding gene-function associations"):
        #     gene = row['Gene']
        #     function = row['Function']
        #     if function not in entities:
        #         ignored_functions += 1
        #         continue
        #     if gene not in entity_to_id:
        #         entity_to_id[gene] = len(entity_to_id)
        #     assert function in entity_to_id, f"Function {function} not in entity_to_id"
        #     query_pattern = "P(Anchor)"
        #     relation = "has_function"
        #     if relation not in relation_to_id:
        #         relation_to_id[relation] = len(relation_to_id)
        #     query = f"P({relation_to_id[relation]},{entity_to_id[gene]})"
        #     answer = entity_to_id[function]
        #     query_to_pattern.append((query, query_pattern))
        #     if query not in query_to_answers:
        #         query_to_answers[query] = set()
        #     query_to_answers[query].add(answer)
        # logger.info(f"Gene-function associations added: {len(gene_functions) - ignored_functions}, Ignored (function not in graph): {ignored_functions}")
        if "has_function" not in relation_to_id:
            relation_to_id["has_function"] = len(relation_to_id)
        logger.info(f"Loaded {len(gene_functions)} gene-function annotations")

    if use_site:
        gene_sites = pd.read_csv("data/gene_site.csv")
        # Individual 1p gene-site queries removed: gene sites are now
        # handled as a group via kip intersection queries. Relation ID still registered.
        # ignored_sites = 0
        # for _, row in tqdm(gene_sites.iterrows(), leave=False, total=len(gene_sites), desc="Adding gene-site associations"):
        #     gene = row['Gene']
        #     site = row['Tissue']
        #     if site not in entities:
        #         ignored_sites += 1
        #         continue
        #     if gene not in entity_to_id:
        #         entity_to_id[gene] = len(entity_to_id)
        #     assert site in entity_to_id, f"Site {site} not in entity_to_id"
        #     query_pattern = "P(Anchor)"
        #     relation = "expressed_in"
        #     if relation not in relation_to_id:
        #         relation_to_id[relation] = len(relation_to_id)
        #     query = f"P({relation_to_id[relation]},{entity_to_id[gene]})"
        #     answer = entity_to_id[site]
        #     query_to_pattern.append((query, query_pattern))
        #     if query not in query_to_answers:
        #         query_to_answers[query] = set()
        #     query_to_answers[query].add(answer)
        # logger.info(f"Gene-site associations added: {len(gene_sites) - ignored_sites}, Ignored (site not in graph): {ignored_sites}")
        if "expressed_in" not in relation_to_id:
            relation_to_id["expressed_in"] = len(relation_to_id)
        logger.info(f"Loaded {len(gene_sites)} gene-site annotations")

    assert len(test_diseases & non_test_diseases) == 0, "Test diseases overlap with train diseases"
    assert len(test_diseases & entities) == 0, "Test diseases overlap with graph diseases"

    for _, row in tqdm(train_disease_genes.iterrows(), leave=False, total=len(train_disease_genes), desc="Adding gene-disease associations"):
        disease = row['Disease']
        gene = row['Gene']
        
        assert gene in entities, f"Gene {gene} not in entities"
        assert disease in entities, f"Disease {disease} not in entities"
            
        query_pattern = "P(Anchor)"

        relation = "associated_with"
        if relation not in relation_to_id:
            relation_to_id[relation] = len(relation_to_id)

        query = f"P({relation_to_id[relation]},{entity_to_id[disease]})"
        answer = entity_to_id[gene]
        query_to_pattern.append((query, query_pattern))
        if query not in query_to_answers:
            query_to_answers[query] = set()
        query_to_answers[query].add(answer)

    gene2pheno = dict()
    if use_phenotypes:
        used_phenos = 0
        ignored_phenos = 0
        for _, row in tqdm(gene_phenotypes.iterrows(), leave=False, total=len(gene_phenotypes), desc="Building gene2pheno mapping"):
            gene = row['Gene']
            phenotype = row['Phenotype']
            if phenotype not in entities:
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
        if phenotype not in entities:
            print(f"Warning: Phenotype {phenotype} for disease {disease} not in graph, ignoring this association")
            ignored_phenos += 1
        else:
            used_phenos += 1
            if disease not in disease2pheno:
                disease2pheno[disease] = []
            disease2pheno[disease].append(phenotype)
    logger.info(f"Disease-Phenotype associations used: {used_phenos}, ignored (phenotype not in graph): {ignored_phenos}. Total diseases with phenotypes: {len(disease2pheno)}")

    gene2function = dict()
    if use_functions:
        ignored_functions = 0
        for _, row in gene_functions.iterrows():
            gene = row['Gene']
            function = row['Function']
            if function not in entities:
                ignored_functions += 1
                continue
            if gene not in gene2function:
                gene2function[gene] = []
            gene2function[gene].append(function)
        logger.info(f"Gene-function associations used for validation/testing: {len(gene_functions) - ignored_functions}, ignored (function not in graph): {ignored_functions}. Total genes with functions: {len(gene2function)}")

    gene2site = dict()
    if use_site:
        for _, row in gene_sites.iterrows():
            gene = row['Gene']
            site = row['Tissue']
            if gene not in gene2site:
                gene2site[gene] = []
            gene2site[gene].append(site)
        
    # Add k-intersection queries: I(P(pheno_1), ..., P(pheno_k)) -> disease
    # Commented out: intersection of boxes not performing well; replaced by kip averaging below.
    # has_symptom_id_ki = relation_to_id.get('has_symptom')
    # if has_symptom_id_ki is not None:
    #     ki_added = 0
    #     for disease, phenos in disease2pheno.items():
    #         if disease in val_diseases or disease in test_diseases:
    #             continue
    #         valid_phenos = [p for p in phenos if p in entity_to_id]
    #         k = len(valid_phenos)
    #         if k < 2:
    #             continue
    #         query_pattern = f"ki_{k}"
    #         if query_pattern not in query_pattern_to_name:
    #             query_pattern_to_name[query_pattern] = "ki"
    #         parts = []
    #         for pheno in valid_phenos:
    #             parts.append(str(has_symptom_id_ki))
    #             parts.append(str(entity_to_id[pheno]))
    #         query = "KI(" + ",".join(parts) + ")"
    #         answer = entity_to_id[disease]
    #         query_to_pattern.append((query, query_pattern))
    #         if query not in query_to_answers:
    #             query_to_answers[query] = set()
    #         query_to_answers[query].add(answer)
    #         ki_added += 1
    #     logger.info(f"Added {ki_added} ki queries for diseases with >=2 phenotypes")
    # else:
    #     logger.warning("'has_symptom' not found in relation_to_id; skipping ki queries")
    has_phenotype_id = relation_to_id['has_phenotype']
    associated_with_id = relation_to_id.get('associated_with')

    # --- Build gene-annotation sources for kip queries ---
    kip_sources = []
    if use_functions and gene2function:
        kip_sources.append(('function', gene2function, relation_to_id['has_function']))
    if use_phenotypes and gene2pheno:
        kip_sources.append(('phenotype', gene2pheno, relation_to_id['has_phenotype']))
    if use_site and gene2site:
        kip_sources.append(('site', gene2site, relation_to_id['expressed_in']))

    # Build disease→genes map for ki2p
    disease_to_train_genes = {}
    for _, row in train_disease_genes.iterrows():
        d, g = row['Disease'], row['Gene']
        disease_to_train_genes.setdefault(d, set()).add(entity_to_id[g])

    # --- Compute global max_k (padded tensor width) over all intersection queries ---
    max_k = 2  # minimum
    for disease, phenos in disease2pheno.items():
        if disease in val_diseases or disease in test_diseases or disease not in entity_to_id:
            continue
        max_k = max(max_k, sum(1 for p in phenos if p in entity_to_id))
    for _, gene2annot, _ in kip_sources:
        for annots in gene2annot.values():
            max_k = max(max_k, sum(1 for a in annots if a in entity_to_id))
    logger.info(f"Global max_k for padded intersection queries: {max_k}")

    PAD_ENTITY = 0  # padding slot; embedding is looked up but masked in attention

    # Register single patterns (no k-suffix) so all queries batch together
    query_pattern_to_name["kipd"] = "kip"
    query_pattern_to_name["kip"]  = "kip"
    query_pattern_to_name["ki2p"] = "ki2p"

    # --- kipd: disease phenotypes → has_phenotype → disease ---
    # Data layout: [k_actual, p_1, ..., p_maxk, has_phenotype_id]
    kipd_added = 0
    for disease, phenos in disease2pheno.items():
        if disease in val_diseases or disease in test_diseases or disease not in entity_to_id:
            continue
        valid_phenos = sorted([entity_to_id[p] for p in phenos if p in entity_to_id])
        if len(valid_phenos) < 2:
            continue
        k = len(valid_phenos)
        padded = valid_phenos + [PAD_ENTITY] * (max_k - k)
        parts = [str(k)] + [str(p) for p in padded] + [str(has_phenotype_id)]
        query = "KIP(" + ",".join(parts) + ")"
        query_to_pattern.append((query, "kipd"))
        query_to_answers.setdefault(query, set()).add(entity_to_id[disease])
        kipd_added += 1
    logger.info(f"Added {kipd_added} kipd queries (phenotype intersection → disease)")

    # --- ki2p: disease phenotypes → has_phenotype → associated_with → gene ---
    # Data layout: [k_actual, p_1, ..., p_maxk, has_phenotype_id, associated_with_id]
    ki2p_added = 0
    if associated_with_id is not None:
        for disease, phenos in disease2pheno.items():
            if disease in val_diseases or disease in test_diseases or disease not in entity_to_id:
                continue
            valid_phenos = sorted([entity_to_id[p] for p in phenos if p in entity_to_id])
            if len(valid_phenos) < 2:
                continue
            gene_ids_for_disease = disease_to_train_genes.get(disease, set())
            if not gene_ids_for_disease:
                continue
            k = len(valid_phenos)
            padded = valid_phenos + [PAD_ENTITY] * (max_k - k)
            parts = [str(k)] + [str(p) for p in padded] + [str(has_phenotype_id), str(associated_with_id)]
            query = "KI2P(" + ",".join(parts) + ")"
            query_to_pattern.append((query, "ki2p"))
            query_to_answers.setdefault(query, set()).update(gene_ids_for_disease)
            ki2p_added += 1
        logger.info(f"Added {ki2p_added} ki2p queries (phenotype intersection → disease → gene)")
    else:
        logger.warning("associated_with not in relation_to_id; skipping ki2p queries")

    # --- kip: gene annotations → relation → gene ---
    # Data layout: [k_actual, a_1, ..., a_maxk, relation_id]
    for source_name, gene2annot, relation_id in kip_sources:
        kip_added = 0
        for gene, annots in gene2annot.items():
            valid_annots = sorted([entity_to_id[a] for a in annots if a in entity_to_id])
            if len(valid_annots) < 2:
                continue
            gene_id = entity_to_id.get(gene)
            if gene_id is None:
                continue
            k = len(valid_annots)
            padded = valid_annots + [PAD_ENTITY] * (max_k - k)
            parts = [str(k)] + [str(a) for a in padded] + [str(relation_id)]
            query = "KIP(" + ",".join(parts) + ")"
            query_to_pattern.append((query, "kip"))
            query_to_answers.setdefault(query, set()).add(gene_id)
            kip_added += 1
        logger.info(f"Added {kip_added} kip queries for gene {source_name}s")

    all_gene_diseases = pd.read_csv("data/gene_diseases.csv")
    eval_genes = set(all_gene_diseases['Gene'].values)
    logger.info(f"Number of evaluation genes: {len(eval_genes)}")
    eval_genes = sorted(list(eval_genes))

    # Restrict negatives to the correct entity type per query pattern.
    gene_ids_array = np.array([entity_to_id[g] for g in eval_genes if g in entity_to_id], dtype=np.int64)
    disease_ids_array = np.array([entity_to_id[d] for d in train_diseases if d in entity_to_id], dtype=np.int64)
    pattern_to_neg_pool = {
        "kip":      gene_ids_array,
        "kipd":     disease_ids_array,
        "ki2p":     gene_ids_array,
        "P(Anchor)": gene_ids_array,   # disease→gene: negatives must be other genes
    }

    dataset = TrainDataset(query_to_pattern, len(entity_to_id), negative_sample_size, query_to_answers,
                           pattern_to_neg_pool=pattern_to_neg_pool)
    logger.info(f"Training dataset size: {len(dataset)} samples")
    train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=3,
        collate_fn=TrainDataset.collate_fn,
    ))

    disjoint_iterator = None
    disjoint_pairs_path = "data/disjoint_pairs.txt"
    if os.path.exists(disjoint_pairs_path):
        disjoint_pairs = []
        with open(disjoint_pairs_path) as f:
            for line in f:
                a, b = line.strip().split("\t")
                if a in entity_to_id and b in entity_to_id:
                    disjoint_pairs.append((entity_to_id[a], entity_to_id[b]))
        if disjoint_pairs:
            logger.info(f"Loaded {len(disjoint_pairs)} disjoint pairs")
            disjoint_iterator = SingledirectionalOneShotIterator(DataLoader(
                DisjointDataset(disjoint_pairs),
                batch_size=batch_size,
                shuffle=True, num_workers=1,
                collate_fn=DisjointDataset.collate_fn,
            ))
        else:
            logger.warning("Disjoint pairs file found but no pairs matched entity_to_id")
    else:
        logger.info("No disjoint_pairs.txt found, skipping disjointness loss")

    file_identifier = f"fold_{fold}_embed_{embedding_dim}_lr_{learning_rate}_gamma_{gamma}_alpha_{alpha}_batch_{batch_size}_pheno_{use_phenotypes}_func_{use_functions}_expr_{use_site}"
    results_output_prefix = f"data/results/kge_results_{file_identifier}"

    save_path = f"runs/fold_{fold}_embed_{embedding_dim}_lr_{learning_rate}_gamma_{gamma}_alpha_{alpha}_batch_{batch_size}_pheno_{use_phenotypes}_func_{use_functions}_expr_{use_site}"


    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    model = GeometrE(len(entity_to_id), len(relation_to_id),
                     embedding_dim, gamma, alpha,
                     query_name_dict=query_pattern_to_name,
                     with_answer_embedding=with_answer_embedding,
                     device=device,
                     gene_entity_ids=gene_ids_array,
                     dispersion_weight=dispersion_weight,
                     max_k=max_k).to(device)

    optimizer = make_optimizer(model, learning_rate)

    warm_up_steps = max_epochs // 10
    save_checkpoint_steps = max_epochs // 10
    log_steps = max_epochs // 10
    
    args = None
    if do_train:
        print("Training the model...")

        training_logs = []
        pbar = tqdm(range(max_epochs))
        for step in pbar:

            log = model.train_step(model, optimizer, train_path_iterator, args, step, disjoint_iterator=disjoint_iterator)
            for metric in log:
                wandb_logger.log({'path_' + metric: log[metric]}, step=step)
            training_logs.append(log)
            pbar.set_postfix(loss=f"{log['loss']:.4f}", pos=f"{log['positive_sample_loss']:.4f}", neg=f"{log['negative_sample_loss']:.4f}", disp=f"{log['dispersion_loss']:.4f}")

            if step >= warm_up_steps:
                learning_rate = learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' % (learning_rate, step))
                optimizer = make_optimizer(model, learning_rate)
                warm_up_steps = warm_up_steps * 1.5
            
            if step % save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'learning_rate': learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, save_path)

            if step % validate_every == 0 and step > 0:
                if do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate_qa_model(model,
                                                          val_disease_genes,
                                                          disease2pheno,
                                                          eval_genes,
                                                          entity_to_id,
                                                          relation_to_id,
                                                          output_file_prefix=results_output_prefix,
                                                          verbose=True
                                                          )

                if do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate_qa_model(model,
                                                         test_disease_genes,
                                                         disease2pheno,
                                                         eval_genes,
                                                         entity_to_id,
                                                         relation_to_id,
                                                         output_file_prefix=results_output_prefix,
                                                         verbose=True
                                                         )
                    
            if step % log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics)
                # Log to wandb
                for metric in metrics:
                    wandb_logger.log({f"train_{metric}": metrics[metric]}, step=step)
                training_logs = []

        save_variable_list = {
            'step': step, 
            'learning_rate': learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, save_path)


def make_optimizer(model, lr):
    # intersection_ids = {id(p) for p in model.intersection_net.parameters()}
    # main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in intersection_ids]
    # inter_params = [p for p in model.intersection_net.parameters() if p.requires_grad]
    # return th.optim.Adam([
    #     {'params': main_params, 'lr': lr},
    #     {'params': inter_params, 'lr': lr * 0.2},
    # ])
    return th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def save_model(model, optimizer, save_variable_list, save_path):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    th.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_path, 'checkpoint')
    )

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
    
        
if __name__ == "__main__":
    main()
