import torch as th
from tqdm import tqdm
from evaluate_sem_sim import compute_metrics, compute_metrics_from_rows, print_as_tex
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def evaluate_by_similarity(model, test_disease_genes, gene2pheno, disease2pheno,
                           eval_genes, triples_factory=None, entity_to_id=None, relation_to_id=None,
                           output_file_prefix=None, verbose=False):
    """
    Evaluate the model using only phenotype embeddings (no relation offsets).

    Builds gene phenotype vectors directly from entity embeddings (no inverse-relation
    offset), then scores each (disease, gene) pair via BMA/BMM over phenotype similarity.

    Args:
        model: The trained KGE model
        test_disease_genes: DataFrame with 'Disease' and 'Gene' columns
        gene2pheno: Dictionary mapping genes to phenotypes
        disease2pheno: Dictionary mapping diseases to phenotypes
        eval_genes: List of genes to evaluate
        triples_factory: Optional PyKEEN triples factory
        entity_to_id: Optional mapping from entity names to IDs
        relation_to_id: Optional mapping from relation names to IDs
        output_file_prefix: Optional prefix for output files. If None, results are not saved.
        verbose: Whether to print detailed results.

    Returns:
        tuple: (inductive_bma_macro_metrics, inductive_bmm_macro_metrics)
    """
    if triples_factory is None and entity_to_id is None and relation_to_id is None:
        raise ValueError("Either triples_factory or both entity_to_id and relation_to_id must be provided.")

    entity_to_id = triples_factory.entity_to_id if triples_factory else entity_to_id

    entity_ids = th.tensor(list(entity_to_id.values()))
    entity_embeddings = model.entity_representations[0](indices=entity_ids).cpu().detach()
    embedding_dim = entity_embeddings.shape[1]

    max_pheno_count = 0
    gene_counts = []
    for gene in eval_genes:
        count = len(gene2pheno.get(gene, []))
        gene_counts.append(count)
        if count > max_pheno_count:
            max_pheno_count = count

    max_pheno_count = max(max_pheno_count, 1)

    all_genes_vectors = th.zeros(len(eval_genes), max_pheno_count, embedding_dim)

    for i, gene in enumerate(eval_genes):
        phenos = gene2pheno.get(gene, [])
        if phenos:
            pheno_ids = [entity_to_id[p] for p in phenos]
            all_genes_vectors[i, :len(pheno_ids), :] = entity_embeddings[th.tensor(pheno_ids)]

    gene_pheno_counts = th.tensor(gene_counts, dtype=th.float32)
    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}

    test_pairs = [(row['Disease'], row['Gene']) for _, row in test_disease_genes.iterrows()]

    inductive_bma_results = []
    inductive_bmm_results = []

    with tqdm(total=len(test_pairs), desc='Evaluating', leave=False) as pbar:
        for test_disease, test_gene in test_pairs:
            disease_phenos = disease2pheno[test_disease]
            pheno_ids = [entity_to_id[p] for p in disease_phenos]
            disease_phenos_vectors = entity_embeddings[th.tensor(pheno_ids)]

            inductive_bma_scores = compare_vectorized(all_genes_vectors, disease_phenos_vectors, gene_pheno_counts, criterion="bma")
            inductive_bmm_scores = compare_vectorized(all_genes_vectors, disease_phenos_vectors, gene_pheno_counts, criterion="bmm")

            inductive_bma_results.append((test_gene, test_disease, gene_to_index[test_gene], inductive_bma_scores.tolist()))
            inductive_bmm_results.append((test_gene, test_disease, gene_to_index[test_gene], inductive_bmm_scores.tolist()))
            pbar.update()

    inductive_bma_macro_metrics = None
    inductive_bmm_macro_metrics = None

    # See evaluate_by_graph: metrics from memory, files only when a prefix is given.
    _, inductive_bma_macro_metrics = compute_metrics_from_rows(inductive_bma_results, verbose=verbose)
    _, inductive_bmm_macro_metrics = compute_metrics_from_rows(inductive_bmm_results, verbose=verbose)

    if output_file_prefix:
        bma_out_file = f"{output_file_prefix}_inductive_bma.tsv"
        bmm_out_file = f"{output_file_prefix}_inductive_bmm.tsv"
        for out_file, results in [(bma_out_file, inductive_bma_results), (bmm_out_file, inductive_bmm_results)]:
            with open(out_file, "w") as f:
                for gene, disease, gene_index, scores in results:
                    f.write(f"{gene}\t{disease}\t{gene_index}\t" + "\t".join(str(s) for s in scores) + "\n")

        if verbose:
            print(f"Inductive results saved to {bma_out_file}")
            print(f"Inductive results saved to {bmm_out_file}")
            print_as_tex(inductive_bma_macro_metrics, "Inductive BMA")
            print_as_tex(inductive_bmm_macro_metrics, "Inductive BMM")

    return (inductive_bma_macro_metrics, inductive_bmm_macro_metrics)



def _calibrated_rows(rows):
    """Leave-one-out per-gene z-scoring of in-memory score rows."""
    import numpy as np
    scores = np.asarray([r[3] for r in rows], dtype=np.float64)
    n = scores.shape[0]
    if n < 3:
        return rows
    mean = (scores.sum(axis=0, keepdims=True) - scores) / (n - 1)
    var = ((scores ** 2).sum(axis=0, keepdims=True) - scores ** 2) / (n - 1) - mean ** 2
    spread = np.sqrt(np.clip(var, 0, None)) + 1e-12
    z = (scores - mean) / spread
    return [(r[0], r[1], r[2], z[i].tolist()) for i, r in enumerate(rows)]


def evaluate_by_graph(model, test_disease_genes, disease2pheno,
                       eval_genes, triples_factory=None, entity_to_id=None, relation_to_id=None,
                       output_file_prefix=None, verbose=False, score_relation_internal=None,
                       calibrate=False):
    """
    Evaluate the 1-hop query via model.predict_hrt:
      gene -[causes_phenotype]-> phenotype

    For each test disease, scores every eval gene against every disease symptom via
    predict_hrt((gene, causes_phenotype, symptom)), giving a (num_genes, num_symptoms)
    score matrix. BMA and BMM are derived as:
      gene_centric[g]    = max  over symptoms  (one vector per gene)
      disease_centric[g] = mean over symptoms
      BMA = (gene_centric + disease_centric) / 2
      BMM = max(gene_centric, disease_centric)

    Args:
        model: Trained TransD model.
        test_disease_genes: DataFrame with 'Disease' and 'Gene' columns.
        disease2pheno: Dict mapping disease URIs to lists of phenotype URIs.
        eval_genes: Ordered list of candidate genes (defines score vector order).
        triples_factory: PyKEEN triples factory (or pass entity_to_id/relation_to_id directly).
        entity_to_id: Entity-name → integer-id mapping.
        relation_to_id: Relation-name → integer-id mapping.
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
        with tqdm(total=len(test_pairs), desc='Evaluating (gene→pheno, predict_hrt)', leave=False) as pbar:
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

                if score_relation_internal is None:
                    scores = model.predict_hrt(hrt).cpu()        # (num_genes * num_symptoms,)
                else:
                    probe = hrt.clone()
                    probe[:, 1] = score_relation_internal
                    scores = model.score_hrt(probe).cpu()
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

    # Metrics are computed from the in-memory rows. The score files are an output
    # artifact, not an input to the metric: writing them and parsing them straight back
    # cost a validation pass ~100 MB of I/O plus ten million float/string conversions,
    # every twenty epochs, to recover one number. BMA is scored before BMM, as before,
    # so the tie-breaking RNG is consumed in the same order and the metrics are unchanged.
    bma_for_metrics = _calibrated_rows(bma_results) if calibrate else bma_results
    bmm_for_metrics = _calibrated_rows(bmm_results) if calibrate else bmm_results

    _, bma_macro_metrics = compute_metrics_from_rows(bma_for_metrics, verbose=verbose)
    _, bmm_macro_metrics = compute_metrics_from_rows(bmm_for_metrics, verbose=verbose)

    if output_file_prefix:
        bma_out_file = f"{output_file_prefix}_by_graph_bma.tsv"
        bmm_out_file = f"{output_file_prefix}_by_graph_bmm.tsv"

        for out_file, results in [(bma_out_file, bma_results), (bmm_out_file, bmm_results)]:
            with open(out_file, "w") as f:
                for gene, disease, gene_index, scores in results:
                    f.write(f"{gene}\t{disease}\t{gene_index}\t" +
                            "\t".join(str(s) for s in scores) + "\n")

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
    
                                                                                        
    
