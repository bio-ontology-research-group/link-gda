import torch as th
from tqdm import tqdm
from evaluate_sem_sim import compute_metrics, print_as_tex
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def evaluate_by_similarity(model, test_disease_genes, gene2pheno,
                           gene2function, gene2site, disease2pheno,
                           eval_genes, triples_factory=None, entity_to_id=None, relation_to_id=None,
                           use_phenotypes=True, use_functions=True, use_site=True,
                           output_file_prefix=None, verbose=False):
    """
    Evaluate the model on a given test set.

    Args:
        model: The trained KGE model
        test_disease_genes: DataFrame with 'Disease' and 'Gene' columns
        gene2pheno: Dictionary mapping genes to phenotypes
        gene2function: Dictionary mapping genes to functions
        gene2site: Dictionary mapping genes to site patterns
        disease2pheno: Dictionary mapping diseases to phenotypes
        eval_genes: List of genes to evaluate
        triples_factory: Optional PyKEEN triples factory (if entity_to_id and relation_to_id are not provided)
        entity_to_id: Optional mapping from entity names to IDs (if triples_factory is not provided)
        relation_to_id: Optional mapping from relation names to IDs (if triples_factory is not provided)
        use_phenotypes: Whether to use gene phenotype information
        use_functions: Whether to use gene function information
        use_site: Whether to use gene site information
        output_file_prefix: Optional prefix for output files. If None, results are not saved.
        verbose: Whether to print detailed results.
        
    Returns:
        tuple: (inductive_bma_macro_metrics, inductive_bmm_macro_metrics)
    """
    if triples_factory is None and entity_to_id is None and relation_to_id is None:
        raise ValueError("Either triples_factory or both entity_ids and relation_ids must be provided.")

    entity_to_id = triples_factory.entity_to_id if triples_factory else entity_to_id
    relation_to_id = triples_factory.relation_to_id if triples_factory else relation_to_id
    entity_ids = th.tensor(list(entity_to_id.values()))
    relation_ids = th.tensor(list(relation_to_id.values()))

    entity_embeddings = model.entity_representations[0](indices=entity_ids).cpu().detach()
    relation_embeddings = model.relation_representations[0](indices=relation_ids).cpu().detach()
    if use_phenotypes:
        has_phenotype_id = relation_to_id['has_phenotype']
        if triples_factory:
            has_phenotype_inverse_id = triples_factory.get_inverse_relation_id(has_phenotype_id)
        else:
            has_phenotype_inverse_id = has_phenotype_id + len(relation_to_id) // 2  # Assuming inverse relations are added in the second half
    
    if use_functions:
        has_function_id = relation_to_id['has_function']
        if triples_factory:
            has_function_inverse_id = triples_factory.get_inverse_relation_id(has_function_id)
        else:
            has_function_inverse_id = has_function_id + len(relation_to_id) // 2
            
    if use_site:
        expressed_in_id = relation_to_id['expressed_in']
        if triples_factory:
            expressed_in_inverse_id = triples_factory.get_inverse_relation_id(expressed_in_id)
        else:
            expressed_in_inverse_id = expressed_in_id + len(relation_to_id) // 2

            
    embedding_dim = entity_embeddings.shape[1]

    logger.debug("Pre-computing gene phenotype vectors...")

    max_pheno_count = 0
    gene_counts = []

    for gene in eval_genes:
        phenos = gene2pheno.get(gene, []) if use_phenotypes else []
        functions = gene2function.get(gene, []) if use_functions else []
        site = gene2site.get(gene, []) if use_site else []
        count = len(phenos) + len(functions) + len(site)
        gene_counts.append(count)
        if count > max_pheno_count:
            max_pheno_count = count

    logger.debug(f"Maximum number of annotations per gene: {max_pheno_count}")
    max_pheno_count = max(max_pheno_count, 1)  # Avoid empty tensor dimension

    all_genes_vectors = th.zeros(len(eval_genes), max_pheno_count, embedding_dim)

    if use_phenotypes:
        inverse_has_phenotype_embedding = relation_embeddings[has_phenotype_inverse_id]
    if use_functions:
        inverse_has_function_embedding = relation_embeddings[has_function_inverse_id]
    if use_site:
        inverse_expressed_in_embedding = relation_embeddings[expressed_in_inverse_id]

    for i, gene in enumerate(eval_genes):
        phenos = gene2pheno.get(gene, []) if use_phenotypes else []
        functions = gene2function.get(gene, []) if use_functions else []
        sites = gene2site.get(gene, []) if use_site else []
        pheno_ids = [entity_to_id[p] for p in phenos] if phenos else None
        function_ids = [entity_to_id[f] for f in functions] if functions else None
        site_ids = [entity_to_id[e] for e in sites] if sites else None

        pheno_vectors = entity_embeddings[th.tensor(pheno_ids)] if pheno_ids else th.zeros(0, embedding_dim)
        function_vectors = entity_embeddings[th.tensor(function_ids)] if function_ids else th.zeros(0, embedding_dim)
        site_vectors = entity_embeddings[th.tensor(site_ids)] if site_ids else th.zeros(0, embedding_dim)
        if use_phenotypes and pheno_ids:
            pheno_vectors = pheno_vectors + inverse_has_phenotype_embedding
        if use_functions and function_ids:
            function_vectors = function_vectors + inverse_has_function_embedding
        if use_site and site_ids:
            site_vectors = site_vectors + inverse_expressed_in_embedding

        offset = 0
        all_genes_vectors[i, offset:offset+len(phenos), :] = pheno_vectors
        offset += len(phenos)
        all_genes_vectors[i, offset:offset+len(functions), :] = function_vectors
        offset += len(functions)
        all_genes_vectors[i, offset:offset+len(sites), :] = site_vectors
        
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
        
    with tqdm(total=len(test_pairs), desc='Evaluating', leave=False) as pbar:
        for test_disease, test_gene in test_pairs:

            disease_phenos = disease2pheno[test_disease]
            pheno_ids = [entity_to_id[p] for p in disease_phenos]

            disease_phenos_vectors = entity_embeddings[th.tensor(pheno_ids)]
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


def evaluate_by_graph(model, test_disease_genes, gene2pheno, gene2function, gene2site, disease2pheno,
                       eval_genes, triples_factory=None, entity_to_id=None, relation_to_id=None,
                       use_phenotypes=True, use_functions=True, use_site=True,
                       output_file_prefix=None, verbose=False):
    """
    Evaluate using model.score_hrt directly for the 1-hop query:
      gene -[causes_phenotype]-> phenotype

    For each test disease, scores every eval gene against every disease symptom via
    score_hrt((gene, causes_phenotype, symptom)), giving a (num_genes, num_symptoms)
    score matrix. BMA and BMM are derived as:
      gene_centric[g]    = max  over symptoms  (one vector per gene)
      disease_centric[g] = mean over symptoms
      BMA = (gene_centric + disease_centric) / 2
      BMM = max(gene_centric, disease_centric)

    Args:
        model: Trained TransD model.
        test_disease_genes: DataFrame with 'Disease' and 'Gene' columns.
        gene2pheno, gene2function, gene2site: Unused; kept for signature compatibility.
        disease2pheno: Dict mapping disease URIs to lists of phenotype URIs.
        eval_genes: Ordered list of candidate genes (defines score vector order).
        triples_factory: PyKEEN triples factory (or pass entity_to_id/relation_to_id directly).
        entity_to_id: Entity-name → integer-id mapping.
        relation_to_id: Relation-name → integer-id mapping.
        use_phenotypes, use_functions, use_site: Unused; kept for signature compatibility.
        output_file_prefix: If set, writes result TSVs and computes metrics.
        verbose: Print detailed results.

    Returns:
        tuple: (bma_macro_metrics, bmm_macro_metrics)
    """
    if triples_factory is None and entity_to_id is None and relation_to_id is None:
        raise ValueError("Either triples_factory or both entity_to_id and relation_to_id must be provided.")

    entity_to_id = triples_factory.entity_to_id if triples_factory else entity_to_id
    relation_to_id = triples_factory.relation_to_id if triples_factory else relation_to_id

    if 'causes_phenotype' not in relation_to_id:
        raise ValueError("Relation 'causes_phenotype' not found in relation_to_id.")

    causes_phenotype_id = relation_to_id['causes_phenotype']
    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}
    num_genes = len(eval_genes)

    gene_entity_ids = th.tensor(
        [entity_to_id[gene] for gene in eval_genes], dtype=th.long
    )

    test_pairs = [(row['Disease'], row['Gene']) for _, row in test_disease_genes.iterrows()]

    bma_results = []
    bmm_results = []

    model.eval()
    with th.no_grad():
        with tqdm(total=len(test_pairs), desc='Evaluating (gene→pheno, score_hrt)', leave=False) as pbar:
            for test_disease, test_gene in test_pairs:
                valid_symptom_ids = [
                    entity_to_id[s]
                    for s in disease2pheno.get(test_disease, [])
                    if s in entity_to_id
                ]

                if not valid_symptom_ids:
                    zero = th.zeros(num_genes)
                    bma_results.append((test_gene, test_disease, gene_to_index[test_gene], zero.tolist()))
                    bmm_results.append((test_gene, test_disease, gene_to_index[test_gene], zero.tolist()))
                    pbar.update()
                    continue

                num_symptoms = len(valid_symptom_ids)
                symptom_ids = th.tensor(valid_symptom_ids, dtype=th.long)

                # Build HRT batch: all (gene, causes_phenotype, symptom) combinations
                # shape: (num_genes * num_symptoms, 3)
                h = gene_entity_ids.unsqueeze(1).expand(num_genes, num_symptoms).reshape(-1)
                r = th.full((num_genes * num_symptoms,), causes_phenotype_id, dtype=th.long)
                t = symptom_ids.unsqueeze(0).expand(num_genes, num_symptoms).reshape(-1)
                hrt = th.stack([h, r, t], dim=1).to(model.device)

                scores = model.score_hrt(hrt).cpu()              # (num_genes * num_symptoms,)
                scores = scores.view(num_genes, num_symptoms)    # (num_genes, num_symptoms)

                gene_centric = scores.max(dim=1).values          # (num_genes,)
                disease_centric = scores.mean(dim=1)             # (num_genes,)

                bma = (gene_centric + disease_centric) / 2
                bmm = th.max(gene_centric, disease_centric)

                bma_results.append((test_gene, test_disease, gene_to_index[test_gene], bma.tolist()))
                bmm_results.append((test_gene, test_disease, gene_to_index[test_gene], bmm.tolist()))
                pbar.update()

    bma_macro_metrics = None
    bmm_macro_metrics = None

    if output_file_prefix:
        bma_out_file = f"{output_file_prefix}_by_graph_bma.tsv"
        bmm_out_file = f"{output_file_prefix}_by_graph_bmm.tsv"

        for out_file, results in [(bma_out_file, bma_results), (bmm_out_file, bmm_results)]:
            with open(out_file, "w") as f:
                for gene, disease, gene_index, scores in results:
                    f.write(f"{gene}\t{disease}\t{gene_index}\t" +
                            "\t".join(str(s) for s in scores) + "\n")

        _, bma_macro_metrics = compute_metrics(bma_out_file, verbose=verbose)
        _, bmm_macro_metrics = compute_metrics(bmm_out_file, verbose=verbose)

        if verbose:
            print(f"Gene-origin results saved to {bma_out_file}")
            print(f"Gene-origin results saved to {bmm_out_file}")
            print_as_tex(bma_macro_metrics, "From-Gene BMA")
            print_as_tex(bmm_macro_metrics, "From-Gene BMM")

    return (bma_macro_metrics, bmm_macro_metrics)


def compare_vectorized(all_genes_pheno_vectors, disease_phenos_vectors, gene_pheno_counts, criterion="bma", similarity="dot",
                       precomputed_a_sq=None, precomputed_pad_mask=None):
    """
    Compute similarity between a disease and all genes in a vectorized manner.

    :param all_genes_pheno_vectors: Padded tensor of shape (num_genes, max_phenos, emb_dim)
    :param disease_phenos_vectors: Tensor of shape (num_disease_phenos, emb_dim)
    :param gene_pheno_counts: Tensor of shape (num_genes,) with counts of real phenotypes for each gene.
    :param criterion: Aggregation criterion ('bma' or 'bmm').
    :param similarity: Similarity function ('dot' for dot-product, 'l2' for negative squared L2 distance).
                       Use 'l2' when vectors come from a TransD projection to match the model's scoring geometry.
    :param precomputed_a_sq: (l2 only) Precomputed squared norms of gene vectors, shape
                             (num_genes * max_phenos, 1).  Pass this to avoid recomputing on every call.
    :param precomputed_pad_mask: (l2 only) Precomputed boolean padding mask, shape (num_genes, max_phenos).
                                 Pass this to avoid recomputing on every call.
    """

    num_genes, max_phenos, emb_dim = all_genes_pheno_vectors.shape
    num_disease_phenos = disease_phenos_vectors.shape[0]

    if similarity == "dot":
        # Reshape for matrix multiplication: (num_genes * max_phenos, d) x (d, num_disease_phenos)
        sim_matrix = th.matmul(
            all_genes_pheno_vectors.view(-1, emb_dim),
            disease_phenos_vectors.T
        )

        # before sigmoid make 0s very negative (padding mask heuristic)
        sim_matrix[sim_matrix == 0] = -th.inf

        sim_matrix = th.sigmoid(sim_matrix)  # (num_genes * max_phenos, num_disease_phenos)
        sim_matrix = sim_matrix.view(num_genes, max_phenos, num_disease_phenos)

    elif similarity == "l2":
        # Negative squared L2 distance — matches TransD's training objective.
        # Computed efficiently via ||a-b||² = ||a||² - 2(a·b) + ||b||²
        # to avoid the huge 4D (num_genes, max_phenos, num_disease_phenos, d) tensor.
        a = all_genes_pheno_vectors.view(-1, emb_dim)          # (num_genes * max_phenos, d)
        b = disease_phenos_vectors                              # (num_disease_phenos, d)
        a_sq = precomputed_a_sq if precomputed_a_sq is not None else a.pow(2).sum(dim=-1, keepdim=True)
        b_sq = b.pow(2).sum(dim=-1).unsqueeze(0)               # (1, num_disease_phenos)
        cross = th.matmul(a, b.T)                              # (num_genes * max_phenos, num_disease_phenos)
        sim_matrix = -(a_sq - 2 * cross + b_sq)               # (num_genes * max_phenos, num_disease_phenos)
        sim_matrix = sim_matrix.view(num_genes, max_phenos, num_disease_phenos)

        # Mask padded slots (indices >= gene count) to -inf so they never win max/mean
        if precomputed_pad_mask is not None:
            pad_mask = precomputed_pad_mask
        else:
            pad_mask = (th.arange(max_phenos, device=all_genes_pheno_vectors.device)
                        .unsqueeze(0) >= gene_pheno_counts.long().unsqueeze(1))  # (num_genes, max_phenos)
        sim_matrix = sim_matrix.masked_fill(pad_mask.unsqueeze(-1), -th.inf)

        sim_matrix = th.sigmoid(sim_matrix)  # (num_genes, max_phenos, num_disease_phenos)

    else:
        raise NotImplementedError(f"Similarity '{similarity}' not implemented. Choose 'dot' or 'l2'.")

    
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
    
                                                                                        
    
