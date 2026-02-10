import torch as th
from tqdm import tqdm
from evaluate_sem_sim import compute_metrics, print_as_tex

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def evaluate_model(model, test_disease_genes, gene2pheno,
                   gene2function, gene2expression, disease2pheno,
                   eval_genes, triples_factory, graph3, graph4,
                   output_file_prefix=None, verbose=False):
    """
    Evaluate the model on a given test set.

    Args:
        model: The trained KGE model
        test_disease_genes: DataFrame with 'Disease' and 'Gene' columns
        gene2pheno: Dictionary mapping genes to phenotypes
        gene2function: Dictionary mapping genes to functions
        gene2expression: Dictionary mapping genes to expression patterns
        disease2pheno: Dictionary mapping diseases to phenotypes
        eval_genes: List of genes to evaluate
        triples_factory: PyKEEN triples factory
        graph3: Boolean indicating if graph3 mode is active
        graph4: Boolean indicating if graph4 mode is active
        output_file_prefix: Optional prefix for output files. If None, results are not saved.

    Returns:
        tuple: (inductive_micro_metrics, inductive_macro_metrics)
    """
    entity_ids = th.tensor(list(triples_factory.entity_to_id.values()))
    relation_ids = th.tensor(list(triples_factory.relation_to_id.values()))
    entity_embeddings = model.entity_representations[0](indices=entity_ids).cpu().detach()
    relation_embeddings = model.relation_representations[0](indices=relation_ids).cpu().detach()
    entity_to_id = triples_factory.entity_to_id
    relation_to_id = triples_factory.relation_to_id
    has_phenotype_id = relation_to_id['has_phenotype']
    has_phenotype_inverse_id = triples_factory.get_inverse_relation_id(has_phenotype_id)
    has_function_id = relation_to_id['has_function']
    has_function_inverse_id = triples_factory.get_inverse_relation_id(has_function_id)
    expressed_in_id = relation_to_id['expressed_in']
    expressed_in_inverse_id = triples_factory.get_inverse_relation_id(expressed_in_id)
    has_symptom_id = relation_to_id['has_symptom']
    has_symptom_inverse_id = triples_factory.get_inverse_relation_id(has_symptom_id)
 
    embedding_dim = entity_embeddings.shape[1]

    logger.debug("Pre-computing gene phenotype vectors...")

    max_pheno_count = 0
    gene_counts = []

    for gene in eval_genes:
        phenos = gene2pheno.get(gene, [])
        functions = gene2function.get(gene, [])
        expression = gene2expression.get(gene, [])
        count = len(phenos) + len(functions) + len(expression)
        gene_counts.append(count)
        if count > max_pheno_count:
            max_pheno_count = count

    logger.debug(f"Maximum number of phenotypes + function + expression per gene: {max_pheno_count}")

    all_genes_vectors = th.zeros(len(eval_genes), max_pheno_count, embedding_dim)

    has_phenotype_embedding = relation_embeddings[has_phenotype_id]
    inverse_has_phenotype_embedding = relation_embeddings[has_phenotype_inverse_id]
    inverse_has_function_embedding = relation_embeddings[has_function_inverse_id]
    inverse_expressed_in_embedding = relation_embeddings[expressed_in_inverse_id]
    for i, gene in enumerate(eval_genes):
        phenos = gene2pheno.get(gene, [])
        functions = gene2function.get(gene, [])
        expressions = gene2expression.get(gene, [])
        pheno_ids = [entity_to_id[p] for p in phenos] if phenos else None
        function_ids = [entity_to_id[f] for f in functions] if functions else None
        expression_ids = [entity_to_id[e] for e in expressions] if expressions else None

        pheno_vectors = entity_embeddings[th.tensor(pheno_ids)] if pheno_ids else th.zeros(0, embedding_dim)
        function_vectors = entity_embeddings[th.tensor(function_ids)] if function_ids else th.zeros(0, embedding_dim)
        expression_vectors = entity_embeddings[th.tensor(expression_ids)] if expression_ids else th.zeros(0, embedding_dim)
        pheno_vectors = pheno_vectors + inverse_has_phenotype_embedding
        function_vectors = function_vectors + inverse_has_function_embedding
        expression_vectors = expression_vectors + inverse_expressed_in_embedding
        all_genes_vectors[i, :len(phenos), :] = pheno_vectors
        all_genes_vectors[i, len(phenos):len(phenos)+len(functions), :] = function_vectors
        all_genes_vectors[i, len(phenos)+len(functions):len(phenos)+len(functions)+len(expressions), :] = expression_vectors
        
    gene_pheno_counts = th.tensor(gene_counts, dtype=th.float32)

    # Create gene indices mapping for faster lookup
    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}
    logger.debug(f"Example gene to index mapping: {list(gene_to_index.items())[:5]}")

    test_pairs = []
    for _, row in test_disease_genes.iterrows():
        disease = row['Disease']
        gene = row['Gene']
        test_pairs.append((disease, gene))

    logger.debug(f"Number of test pairs: {len(test_pairs)}")
    logger.debug(f"Example test pair: {test_pairs[0]}")

    inductive_bma_results = []
    inductive_bmm_results = []
        
    has_symptom_embedding = relation_embeddings[has_symptom_id]
    inverse_has_symptom_embedding = relation_embeddings[has_symptom_inverse_id]
    with tqdm(total=len(test_pairs), desc='Evaluating', leave=False) as pbar:
        for test_disease, test_gene in test_pairs:

            disease_phenos = disease2pheno[test_disease]
            pheno_ids = [entity_to_id[p] for p in disease_phenos]

            disease_phenos_vectors = entity_embeddings[th.tensor(pheno_ids)]
            # disease_phenos_vectors = disease_phenos_vectors - has_symptom_embedding
            # disease_phenos_vectors = disease_phenos_vectors + inverse_has_symptom_embedding
            inductive_bma_scores = compare_vectorized(all_genes_vectors, disease_phenos_vectors, gene_pheno_counts, criterion="bma")
            inductive_bmm_scores = compare_vectorized(all_genes_vectors, disease_phenos_vectors, gene_pheno_counts, criterion="bmm")
            
            assert inductive_bma_scores.shape == (len(eval_genes),), f"Scores shape {inductive_bma_scores.shape} does not match number of genes {len(eval_genes)}"
            assert inductive_bmm_scores.shape == (len(eval_genes),), f"Scores shape {inductive_bmm_scores.shape} does not match number of genes {len(eval_genes)}"
            inductive_bma_scores = inductive_bma_scores.tolist()
            inductive_bmm_scores = inductive_bmm_scores.tolist()

            inductive_bma_results.append((test_gene, test_disease, gene_to_index[test_gene], inductive_bma_scores))
            inductive_bmm_results.append((test_gene, test_disease, gene_to_index[test_gene], inductive_bmm_scores))

            pbar.update()

    # Compute metrics
    inductive_bma_macro_metrics = None
    inductive_bmm_macro_metrics = None
        
    if output_file_prefix:
        inductive_bma_results_out_file = f"{output_file_prefix}_inductive_bma.tsv"
        inductive_bmm_results_out_file = f"{output_file_prefix}_inductive_bmm.tsv"
        with open(inductive_bma_results_out_file, "w") as f:
            for gene, disease, gene_index, scores in inductive_bma_results:
                scores_str = "\t".join([str(score) for score in scores])
                f.write(f"{gene}\t{disease}\t{gene_index}\t{scores_str}\n")
        with open(inductive_bmm_results_out_file, "w") as f:
            for gene, disease, gene_index, scores in inductive_bmm_results:
                scores_str = "\t".join([str(score) for score in scores])
                f.write(f"{gene}\t{disease}\t{gene_index}\t{scores_str}\n")
                
        _, inductive_bma_macro_metrics = compute_metrics(inductive_bma_results_out_file, verbose=verbose)
        _, inductive_bmm_macro_metrics = compute_metrics(inductive_bmm_results_out_file, verbose=verbose)
        
        if verbose:
            print(f"Inductive results saved to {inductive_bma_results_out_file}")
            print(f"Inductive results saved to {inductive_bmm_results_out_file}")
            print_as_tex(inductive_bma_macro_metrics, "Inductive BMA")
            print_as_tex(inductive_bmm_macro_metrics, "Inductive BMM")

    return (inductive_bma_macro_metrics,
            inductive_bmm_macro_metrics,
            )


def compare_vectorized(all_genes_pheno_vectors, disease_phenos_vectors, gene_pheno_counts, criterion="bma"):
    """
    Compute similarity between a disease and all genes in a vectorized manner.

    :param all_genes_pheno_vectors: Padded tensor of shape (num_genes, max_phenos, emb_dim)
    :param disease_phenos_vectors: Tensor of shape (num_disease_phenos, emb_dim)
    :param gene_pheno_counts: Tensor of shape (num_genes, 1) with counts of real phenotypes for each gene.
    :param criterion: Similarity criterion.
    """

    num_genes, max_phenos, emb_dim = all_genes_pheno_vectors.shape
    num_disease_phenos = disease_phenos_vectors.shape[0]

    # Reshape for matrix multiplication: (num_genes * max_phenos, a) x (a,b)
    sim_matrix = th.matmul(
        all_genes_pheno_vectors.view(-1, emb_dim),
        disease_phenos_vectors.T
    )

    # before sigmoid make 0s very negative
    sim_matrix[sim_matrix == 0] = -th.inf

    sim_matrix = th.sigmoid(sim_matrix) # Resulting shape: (num_genes*max_phenos, num_disease_phenos)

    sim_matrix = sim_matrix.view(num_genes, max_phenos, num_disease_phenos)

    
    # Gene-centric scores
    logger.debug(f"Sim matrix shape: {sim_matrix.shape}")
    gene_max_sim, _ = sim_matrix.max(dim=-1)
    logger.debug(f"Gene max sim shape: {gene_max_sim.shape}")
    gene_centric_sum = gene_max_sim.sum(dim=-1)
    logger.debug(f"Gene centric sum shape: {gene_centric_sum.shape}")
    # For genes with 0 phenotypes, gene_pheno_counts is 0, this will result in NaN. Avoid division by zero.
    # We replace 0 counts with 1 to avoid division by zero. The sum is 0 so the score will be 0.
    gene_pheno_counts_safe = th.max(gene_pheno_counts, th.tensor(1.0))
    logger.debug(f"Gene pheno counts shape: {gene_pheno_counts_safe.shape}")
    gene_centric_scores = gene_centric_sum / gene_pheno_counts_safe
    logger.debug(f"Gene centric scores shape: {gene_centric_scores.shape}")

    assert th.all(gene_centric_scores >= 0) and th.all(gene_centric_scores <= 1), "Gene centric scores out of range [0, 1]"

    # Disease-centric scores
    disease_max_sim, _ = sim_matrix.max(dim=1)
    disease_centric_scores = disease_max_sim.mean(dim=-1)
    # disease_centric_scores = disease_centric_sum / num_disease_phenos

    assert th.all(disease_centric_scores >= 0) and th.all(disease_centric_scores <= 1), "Disease centric scores out of range [0, 1]"
    assert gene_centric_scores.shape == disease_centric_scores.shape == (num_genes,), f"Scores shape mismatch: {gene_centric_scores.shape}, {disease_centric_scores.shape}, expected {(num_genes,)}"
    if criterion == "bma":
        scores = (gene_centric_scores + disease_centric_scores) / 2
    elif criterion == "bmm":
        scores = th.max(gene_centric_scores, disease_centric_scores)

    else:
        raise NotImplementedError(f"Criterion {criterion} not implemented.")

    return scores
    
                                                                                        
    
