import torch as th
from tqdm import tqdm
from evaluate_sem_sim import compute_metrics, print_as_tex
from pykeen.models import TransD
from geometre.embeddings import embedding_2p as _embedding_2p
from geometre.embeddings import embedding_ki2p as _embedding_ki2p
from geometre.box import Box


def transd_project(model: TransD, entity_ids: th.LongTensor, relation_id: int, as_tail: bool = False) -> th.Tensor:
    """
    Project entities through a TransD relation.

    as_tail=False (default): head side — returns h_⊥ + r
    as_tail=True:            tail side — returns h_⊥  (without adding r)
    """
    entity_embs = model.entity_representations[0]()
    entity_projs = model.entity_representations[1]()
    relation_embs = model.relation_representations[0]()
    relation_projs = model.relation_representations[1]()

    h = entity_embs[entity_ids]       # (n, d_e)
    h_p = entity_projs[entity_ids]    # (n, d_e)
    r = relation_embs[relation_id]    # (d_r,)
    r_p = relation_projs[relation_id] # (d_r,)

    dot = (h_p * h).sum(dim=-1, keepdim=True)  # (n, 1)
    h_projected = h + dot * r_p.unsqueeze(0)

    if as_tail:
        return h_projected          # (n, d_r)
    return h_projected + r          # (n, d_r)

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def evaluate_model(model, test_disease_genes, gene2pheno,
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

def evaluate_by_graph_old(model, test_disease_genes, gene2pheno, gene2function, gene2site, disease2pheno,
                       eval_genes, triples_factory=None, entity_to_id=None, relation_to_id=None,
                       use_phenotypes=True, use_functions=True, use_site=True,
                       output_file_prefix=None, verbose=False):
    """
    Evaluate the model using TransD projections through the indirect relations
    (indirect_phenotype_association, indirect_function_association, indirect_site_association).

    For each active relation type, precomputes head-side projected vectors for all gene
    annotations (annot_⊥ + r) and, per disease, projects its symptoms as tails (symptom_⊥).
    compare_vectorized then gives BMA/BMM scores per relation; final score is the mean
    across active relation types.

    This avoids the combinatorial explosion of score_hrt while still using the model's
    learned TransD geometry for each indirect relation.

    Args:
        model: The trained TransD model.
        test_disease_genes: DataFrame with 'Disease' and 'Gene' columns.
        gene2pheno: Dict mapping genes to phenotype entity URIs.
        gene2function: Dict mapping genes to function entity URIs.
        gene2site: Dict mapping genes to site entity URIs.
        disease2pheno: Dict mapping diseases to symptom entity URIs.
        eval_genes: Ordered list of genes to evaluate (defines score vector order).
        triples_factory: PyKEEN triples factory (or pass entity_to_id/relation_to_id directly).
        entity_to_id: Entity-name → integer-id mapping.
        relation_to_id: Relation-name → integer-id mapping.
        use_phenotypes: Include indirect_phenotype_association relation.
        use_functions: Include indirect_function_association relation.
        use_site: Include indirect_site_association relation.
        output_file_prefix: If set, writes result TSVs and computes metrics.
        verbose: Print detailed results.

    Returns:
        tuple: (inductive_bma_macro_metrics, inductive_bmm_macro_metrics)
    """
    if triples_factory is None and entity_to_id is None and relation_to_id is None:
        raise ValueError("Either triples_factory or both entity_to_id and relation_to_id must be provided.")

    entity_to_id = triples_factory.entity_to_id if triples_factory else entity_to_id
    relation_to_id = triples_factory.relation_to_id if triples_factory else relation_to_id

    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}
    num_genes = len(eval_genes)

    # Determine which indirect relations are active and available
    relation_configs = []
    if use_phenotypes and 'indirect_phenotype_association' in relation_to_id:
        relation_configs.append(('pheno', 'indirect_phenotype_association', gene2pheno))
    if use_functions and 'indirect_function_association' in relation_to_id:
        relation_configs.append(('func', 'indirect_function_association', gene2function))
    if use_site and 'indirect_site_association' in relation_to_id:
        relation_configs.append(('site', 'indirect_site_association', gene2site))

    if not relation_configs:
        raise ValueError("No active indirect relations found in the graph. "
                         "Ensure the model was trained with at least one of: "
                         "indirect_phenotype_association, indirect_function_association, indirect_site_association.")

    model.eval()
    with th.no_grad():
        # ------------------------------------------------------------------
        # Precompute gene annotation head-side projected vectors per relation.
        # One batch call to transd_project per relation type.
        # gene_annot_matrices[rel_name] = (matrix: (num_genes, max_count, d_r),
        #                                   counts: (num_genes,))
        # ------------------------------------------------------------------
        gene_annot_matrices = {}

        for rel_name, rel_str, gene2annot in relation_configs:
            rel_id = relation_to_id[rel_str]

            # Collect valid annotation IDs per gene
            gene_annot_ids = []
            for gene in eval_genes:
                ids = [entity_to_id[a] for a in gene2annot.get(gene, []) if a in entity_to_id]
                gene_annot_ids.append(ids)

            counts = [len(ids) for ids in gene_annot_ids]
            max_count = max(counts) if any(counts) else 0
            max_count = max(max_count, 1)  # avoid empty tensor dimension

            # Batch-project all annotations at once
            all_ids_flat = [aid for ids in gene_annot_ids for aid in ids]
            if all_ids_flat:
                all_projected = transd_project(model, th.tensor(all_ids_flat), rel_id, as_tail=False).cpu()
                d_r = all_projected.shape[-1]
            else:
                # No annotations at all — get d_r from a dummy call
                d_r = transd_project(model, th.tensor([0]), rel_id, as_tail=False).cpu().shape[-1]

            matrix = th.zeros(num_genes, max_count, d_r)
            ptr = 0
            for i, ids in enumerate(gene_annot_ids):
                n = len(ids)
                if n:
                    matrix[i, :n, :] = all_projected[ptr:ptr + n]
                    ptr += n

            counts_tensor = th.tensor(counts, dtype=th.float32)
            # Precompute a_sq and pad_mask once — reused for every disease in the scoring loop.
            a_flat = matrix.view(-1, d_r)
            a_sq = a_flat.pow(2).sum(dim=-1, keepdim=True)  # (num_genes * max_count, 1)
            pad_mask = (th.arange(max_count).unsqueeze(0) >= counts_tensor.long().unsqueeze(1))  # (num_genes, max_count)

            gene_annot_matrices[rel_name] = (matrix, counts_tensor, a_sq, pad_mask)
            logger.info(f"[KGE eval] '{rel_str}': {sum(counts)} annotation vectors precomputed "
                        f"for {num_genes} genes (max {max_count} per gene, d_r={d_r})")

        # ------------------------------------------------------------------
        # Precompute tail-side projected disease symptom vectors per relation.
        # Collect all unique symptoms across test diseases, project once per
        # relation, then cache by symptom id for fast per-disease lookup.
        # ------------------------------------------------------------------
        test_pairs = [(row['Disease'], row['Gene']) for _, row in test_disease_genes.iterrows()]
        test_diseases = set(d for d, _ in test_pairs)

        # Unique symptoms that appear in test diseases and exist in the graph
        unique_symptom_ids = sorted({
            entity_to_id[s]
            for disease in test_diseases
            for s in disease2pheno.get(disease, [])
            if s in entity_to_id
        })

        # symptom_tail_vecs[rel_name][symptom_id] -> projected vector (d_r,)
        symptom_tail_vecs = {}
        if unique_symptom_ids:
            symptom_id_tensor = th.tensor(unique_symptom_ids)
            for rel_name, rel_str, _ in relation_configs:
                rel_id = relation_to_id[rel_str]
                projected = transd_project(model, symptom_id_tensor, rel_id, as_tail=True).cpu()
                # shape: (num_unique_symptoms, d_r)
                symptom_tail_vecs[rel_name] = {
                    sid: projected[j] for j, sid in enumerate(unique_symptom_ids)
                }

        # ------------------------------------------------------------------
        # Score every (disease, gene) test pair
        # ------------------------------------------------------------------
        bma_results = []
        bmm_results = []

        with tqdm(total=len(test_pairs), desc='Evaluating (KGE)', leave=False) as pbar:
            for test_disease, test_gene in test_pairs:
                valid_symptom_ids = [
                    entity_to_id[s]
                    for s in disease2pheno.get(test_disease, [])
                    if s in entity_to_id
                ]

                if not valid_symptom_ids or not symptom_tail_vecs:
                    zero = th.zeros(num_genes)
                    bma_results.append((test_gene, test_disease, gene_to_index[test_gene], zero.tolist()))
                    bmm_results.append((test_gene, test_disease, gene_to_index[test_gene], zero.tolist()))
                    pbar.update()
                    continue

                rel_bma_scores = []
                rel_bmm_scores = []

                for rel_name, _, _ in relation_configs:
                    matrix, counts, a_sq, pad_mask = gene_annot_matrices[rel_name]
                    # Stack precomputed tail vectors for this disease's symptoms
                    symptom_vecs = th.stack([symptom_tail_vecs[rel_name][sid] for sid in valid_symptom_ids])
                    # shape: (num_symptoms, d_r)

                    bma = compare_vectorized(matrix, symptom_vecs, counts, criterion="bma", similarity="l2",
                                            precomputed_a_sq=a_sq, precomputed_pad_mask=pad_mask)
                    bmm = compare_vectorized(matrix, symptom_vecs, counts, criterion="bmm", similarity="l2",
                                            precomputed_a_sq=a_sq, precomputed_pad_mask=pad_mask)
                    rel_bma_scores.append(bma)
                    rel_bmm_scores.append(bmm)

                # Average across active relation types → (num_genes,)
                final_bma = th.stack(rel_bma_scores).mean(dim=0)
                final_bmm = th.stack(rel_bmm_scores).mean(dim=0)

                bma_results.append((test_gene, test_disease, gene_to_index[test_gene], final_bma.tolist()))
                bmm_results.append((test_gene, test_disease, gene_to_index[test_gene], final_bmm.tolist()))
                pbar.update()

    # ------------------------------------------------------------------
    # Persist results and compute metrics
    # ------------------------------------------------------------------
    inductive_bma_macro_metrics = None
    inductive_bmm_macro_metrics = None

    if output_file_prefix:
        inductive_bma_results_out_file = f"{output_file_prefix}_inductive_bma.tsv"
        inductive_bmm_results_out_file = f"{output_file_prefix}_inductive_bmm.tsv"

        for out_file, results in [
            (inductive_bma_results_out_file, bma_results),
            (inductive_bmm_results_out_file, bmm_results),
        ]:
            with open(out_file, "w") as f:
                for gene, disease, gene_index, scores in results:
                    f.write(f"{gene}\t{disease}\t{gene_index}\t" +
                            "\t".join(str(s) for s in scores) + "\n")

        _, inductive_bma_macro_metrics = compute_metrics(inductive_bma_results_out_file, verbose=verbose)
        _, inductive_bmm_macro_metrics = compute_metrics(inductive_bmm_results_out_file, verbose=verbose)

        if verbose:
            print(f"Inductive results saved to {inductive_bma_results_out_file}")
            print(f"Inductive results saved to {inductive_bmm_results_out_file}")
            print_as_tex(inductive_bma_macro_metrics, "Inductive BMA (KGE)")
            print_as_tex(inductive_bmm_macro_metrics, "Inductive BMM (KGE)")

    return (inductive_bma_macro_metrics, inductive_bmm_macro_metrics)


def evaluate_by_graph(model, test_disease_genes, gene2pheno, gene2function, gene2site, disease2pheno,
                            eval_genes, triples_factory=None, entity_to_id=None, relation_to_id=None,
                            use_phenotypes=True, use_functions=True, use_site=True,
                            output_file_prefix=None, verbose=False):
    """
    Evaluate the model using 3-hop TransD path queries:

      pheno    -[has_phenotype_inv]-> gene -[associated_with]-> disease -[has_symptom]-> disease_pheno
      function -[has_function_inv]->  gene -[associated_with]-> disease -[has_symptom]-> disease_pheno
      site     -[expressed_in_inv]->  gene -[associated_with]-> disease -[has_symptom]-> disease_pheno

    Hop 1 uses the full TransD projection (transd_project) for the annotation entity.
    Hops 2 and 3 add the corresponding relation embedding vectors (TransE-style path composition).
    Disease phenotypes are projected as tails of has_symptom.

    Args:
        model: The trained TransD model.
        test_disease_genes: DataFrame with 'Disease' and 'Gene' columns.
        gene2pheno: Dict mapping genes to phenotype entity URIs.
        gene2function: Dict mapping genes to function entity URIs.
        gene2site: Dict mapping genes to site entity URIs.
        disease2pheno: Dict mapping diseases to symptom entity URIs.
        eval_genes: Ordered list of genes to evaluate (defines score vector order).
        triples_factory: PyKEEN triples factory (or pass entity_to_id/relation_to_id directly).
        entity_to_id: Entity-name → integer-id mapping.
        relation_to_id: Relation-name → integer-id mapping.
        use_phenotypes: Include the pheno path.
        use_functions: Include the function path.
        use_site: Include the site path.
        output_file_prefix: If set, writes result TSVs and computes metrics.
        verbose: Print detailed results.

    Returns:
        tuple: (inductive_bma_macro_metrics, inductive_bmm_macro_metrics)
    """
    if triples_factory is None and entity_to_id is None and relation_to_id is None:
        raise ValueError("Either triples_factory or both entity_to_id and relation_to_id must be provided.")

    entity_to_id = triples_factory.entity_to_id if triples_factory else entity_to_id
    relation_to_id = triples_factory.relation_to_id if triples_factory else relation_to_id

    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}
    num_genes = len(eval_genes)

    for rel in ('associated_with', 'has_symptom'):
        if rel not in relation_to_id:
            raise ValueError(f"Relation '{rel}' not found in relation_to_id.")

    associated_with_id = relation_to_id['associated_with']
    has_symptom_id = relation_to_id['has_symptom']

    def get_inverse_id(rel_name):
        rel_id = relation_to_id[rel_name]
        return triples_factory.get_inverse_relation_id(rel_id)
        # return rel_id + len(relation_to_id) // 2

    # (name, first-hop inverse relation id, gene->annotation dict)
    relation_configs = []
    if use_phenotypes and 'has_phenotype' in relation_to_id:
        relation_configs.append(('pheno', get_inverse_id('has_phenotype'), gene2pheno))
    if use_functions and 'has_function' in relation_to_id:
        relation_configs.append(('func', get_inverse_id('has_function'), gene2function))
    if use_site and 'expressed_in' in relation_to_id:
        relation_configs.append(('site', get_inverse_id('expressed_in'), gene2site))

    if not relation_configs:
        raise ValueError("No active relations found for 3-hop evaluation.")

    model.eval()
    with th.no_grad():
        # Relation embeddings for hops 2 and 3 (shared across all annotation types)
        relation_embs = model.relation_representations[0]()
        associated_with_emb = relation_embs[associated_with_id].cpu()  # (d_r,)
        has_symptom_emb = relation_embs[has_symptom_id].cpu()          # (d_r,)

        # ------------------------------------------------------------------
        # Precompute 3-hop projected vectors for gene annotations.
        # For annotation a: transd_project(a, r1_inv) + associated_with + has_symptom
        # ------------------------------------------------------------------
        gene_annot_matrices = {}

        for rel_name, r1_inv_id, gene2annot in relation_configs:
            gene_annot_ids = []
            for gene in eval_genes:
                ids = [entity_to_id[a] for a in gene2annot.get(gene, []) if a in entity_to_id]
                gene_annot_ids.append(ids)

            counts = [len(ids) for ids in gene_annot_ids]
            max_count = max(counts) if any(counts) else 0
            max_count = max(max_count, 1)

            all_ids_flat = [aid for ids in gene_annot_ids for aid in ids]
            if all_ids_flat:
                hop1 = transd_project(model, th.tensor(all_ids_flat), r1_inv_id, as_tail=False).cpu()
                all_projected = hop1 + associated_with_emb + has_symptom_emb
                d_r = all_projected.shape[-1]
            else:
                d_r = transd_project(model, th.tensor([0]), r1_inv_id, as_tail=False).cpu().shape[-1]
                all_projected = th.zeros(0, d_r)

            matrix = th.zeros(num_genes, max_count, d_r)
            ptr = 0
            for i, ids in enumerate(gene_annot_ids):
                n = len(ids)
                if n:
                    matrix[i, :n, :] = all_projected[ptr:ptr + n]
                    ptr += n

            counts_tensor = th.tensor(counts, dtype=th.float32)
            a_flat = matrix.view(-1, d_r)
            a_sq = a_flat.pow(2).sum(dim=-1, keepdim=True)
            pad_mask = (th.arange(max_count).unsqueeze(0) >= counts_tensor.long().unsqueeze(1))

            gene_annot_matrices[rel_name] = (matrix, counts_tensor, a_sq, pad_mask)
            logger.info(f"[3-hop eval] '{rel_name}': {sum(counts)} annotation vectors precomputed "
                        f"for {num_genes} genes (max {max_count} per gene, d_r={d_r})")

        # ------------------------------------------------------------------
        # Precompute disease symptom tail projections through has_symptom.
        # All annotation types share the same tail projection.
        # ------------------------------------------------------------------
        test_pairs = [(row['Disease'], row['Gene']) for _, row in test_disease_genes.iterrows()]
        test_diseases = set(d for d, _ in test_pairs)

        unique_symptom_ids = sorted({
            entity_to_id[s]
            for disease in test_diseases
            for s in disease2pheno.get(disease, [])
            if s in entity_to_id
        })

        symptom_tail_vecs = {}
        if unique_symptom_ids:
            symptom_id_tensor = th.tensor(unique_symptom_ids)
            projected = transd_project(model, symptom_id_tensor, has_symptom_id, as_tail=True).cpu()
            symptom_tail_vecs = {sid: projected[j] for j, sid in enumerate(unique_symptom_ids)}

        # ------------------------------------------------------------------
        # Score every (disease, gene) test pair
        # ------------------------------------------------------------------
        bma_results = []
        bmm_results = []

        with tqdm(total=len(test_pairs), desc='Evaluating (3-hop)', leave=False) as pbar:
            for test_disease, test_gene in test_pairs:
                valid_symptom_ids = [
                    entity_to_id[s]
                    for s in disease2pheno.get(test_disease, [])
                    if s in entity_to_id
                ]

                if not valid_symptom_ids or not symptom_tail_vecs:
                    zero = th.zeros(num_genes)
                    bma_results.append((test_gene, test_disease, gene_to_index[test_gene], zero.tolist()))
                    bmm_results.append((test_gene, test_disease, gene_to_index[test_gene], zero.tolist()))
                    pbar.update()
                    continue

                symptom_vecs = th.stack([symptom_tail_vecs[sid] for sid in valid_symptom_ids])

                rel_bma_scores = []
                rel_bmm_scores = []

                for rel_name, _, _ in relation_configs:
                    matrix, counts, a_sq, pad_mask = gene_annot_matrices[rel_name]

                    bma = compare_vectorized(matrix, symptom_vecs, counts, criterion="bma", similarity="l2",
                                            precomputed_a_sq=a_sq, precomputed_pad_mask=pad_mask)
                    bmm = compare_vectorized(matrix, symptom_vecs, counts, criterion="bmm", similarity="l2",
                                            precomputed_a_sq=a_sq, precomputed_pad_mask=pad_mask)
                    rel_bma_scores.append(bma)
                    rel_bmm_scores.append(bmm)

                final_bma = th.stack(rel_bma_scores).mean(dim=0)
                final_bmm = th.stack(rel_bmm_scores).mean(dim=0)

                bma_results.append((test_gene, test_disease, gene_to_index[test_gene], final_bma.tolist()))
                bmm_results.append((test_gene, test_disease, gene_to_index[test_gene], final_bmm.tolist()))
                pbar.update()

    # ------------------------------------------------------------------
    # Persist results and compute metrics
    # ------------------------------------------------------------------
    inductive_bma_macro_metrics = None
    inductive_bmm_macro_metrics = None

    if output_file_prefix:
        inductive_bma_results_out_file = f"{output_file_prefix}_inductive_bma.tsv"
        inductive_bmm_results_out_file = f"{output_file_prefix}_inductive_bmm.tsv"

        for out_file, results in [
            (inductive_bma_results_out_file, bma_results),
            (inductive_bmm_results_out_file, bmm_results),
        ]:
            with open(out_file, "w") as f:
                for gene, disease, gene_index, scores in results:
                    f.write(f"{gene}\t{disease}\t{gene_index}\t" +
                            "\t".join(str(s) for s in scores) + "\n")

        _, inductive_bma_macro_metrics = compute_metrics(inductive_bma_results_out_file, verbose=verbose)
        _, inductive_bmm_macro_metrics = compute_metrics(inductive_bmm_results_out_file, verbose=verbose)

        if verbose:
            print(f"Inductive results saved to {inductive_bma_results_out_file}")
            print(f"Inductive results saved to {inductive_bmm_results_out_file}")
            print_as_tex(inductive_bma_macro_metrics, "Inductive BMA (3-hop)")
            print_as_tex(inductive_bmm_macro_metrics, "Inductive BMM (3-hop)")

    return (inductive_bma_macro_metrics, inductive_bmm_macro_metrics)


def evaluate_qa_model(model, test_disease_genes, disease2pheno, eval_genes,
                      entity_to_id, relation_to_id,
                      output_file_prefix=None, verbose=False):
    """
    Evaluate the GeometrE QA model using the backward 2-hop direction.

    Inference chain per disease phenotype:
      pheno -[has_symptom]-> disease -[associated_with]-> gene

    Steps:
      1. For each disease phenotype compute a gene query box via embedding_2p.
      2. Score all eval genes by L1 distance between gene center and query box center.
      3. Aggregate scores across phenotypes as (mean + max) / 2.

    Args:
        model: Trained GeometrE model.
        test_disease_genes: DataFrame with 'Disease' and 'Gene' columns.
        disease2pheno: Dict mapping disease URIs to lists of phenotype URIs.
        eval_genes: Ordered list of candidate genes (defines score vector order).
        entity_to_id: Entity-name -> int ID mapping.
        relation_to_id: Relation-name -> int ID mapping.
        output_file_prefix: If set, writes a result TSV and computes ranking metrics.
        verbose: Print detailed results.

    Returns:
        macro_metrics dict (or None if output_file_prefix is not set).
    """
    for rel in ('has_symptom', 'associated_with'):
        if rel not in relation_to_id:
            raise ValueError(f"Relation '{rel}' not found in relation_to_id.")

    has_symptom_id = relation_to_id['has_symptom']
    associated_with_id = relation_to_id['associated_with']

    gene_to_index = {gene: i for i, gene in enumerate(eval_genes)}
    eval_gene_ids = th.tensor(
        [entity_to_id[gene] for gene in eval_genes], dtype=th.long
    ).to(model.device)

    test_pairs = [(row['Disease'], row['Gene']) for _, row in test_disease_genes.iterrows()]

    results = []
    model.eval()
    with th.no_grad():
        with tqdm(total=len(test_pairs), desc='Evaluating (QA)', leave=False) as pbar:
            for test_disease, test_gene in test_pairs:
                disease_phenos = disease2pheno.get(test_disease, [])
                valid_phenos = [p for p in disease_phenos if p in entity_to_id]

                if not valid_phenos:
                    scores = th.zeros(len(eval_genes))
                    results.append((test_gene, test_disease, gene_to_index[test_gene], scores.tolist()))
                    pbar.update()
                    continue

                # Intersect all disease phenotype boxes, project through has_symptom
                # to the disease region, then through associated_with to gene space.
                # Data layout for embedding_ki2p: [e_1, ..., e_k, r_has_symptom, r_associated_with]
                pheno_entity_ids = [entity_to_id[p] for p in valid_phenos]
                data = th.tensor(
                    [pheno_entity_ids + [has_symptom_id, associated_with_id]],
                    dtype=th.long, device=model.device
                )
                gene_query_box, *_ = _embedding_ki2p(
                    data, model.get_box_data(), model.get_role_data(),
                    model.transitive_ids, model.inverse_ids, False,
                    intersection_net=model.get_intersection_net()
                )
                # gene_query_box: center/offset shape (1, dim) — the predicted gene region

                # Score all eval genes via box inclusion score.
                if model.with_answer_embedding:
                    gene_centers = model.answer_embedding(eval_gene_ids)       # (n_genes, dim)
                    gene_offsets = th.zeros_like(gene_centers)
                else:
                    gene_centers = model.center_embedding(eval_gene_ids)       # (n_genes, dim)
                    gene_offsets = th.abs(model.offset_embedding(eval_gene_ids))

                # Add a middle dim so box_inclusion_score norms over the last dim only.
                gene_boxes  = Box(gene_centers.unsqueeze(1), gene_offsets.unsqueeze(1))  # (n_genes, 1, dim)
                query_box   = Box(gene_query_box.center.unsqueeze(0),
                                  gene_query_box.offset.unsqueeze(0))                    # (1, 1, dim)

                inclusion = Box.box_inclusion_score(query_box, gene_boxes, model.alpha)  # (n_genes, 1)
                scores = -(model.gamma - inclusion).squeeze(1).cpu()                       # (n_genes,)
                
                results.append((test_gene, test_disease, gene_to_index[test_gene], scores.tolist()))
                pbar.update()

    macro_metrics = None
    if output_file_prefix:
        out_file = f"{output_file_prefix}_qa.tsv"
        with open(out_file, "w") as f:
            for gene, disease, gene_index, scores in results:
                f.write(f"{gene}\t{disease}\t{gene_index}\t" +
                        "\t".join(str(s) for s in scores) + "\n")

        _, macro_metrics = compute_metrics(out_file, verbose=verbose)

        if verbose:
            print(f"QA results saved to {out_file}")
            print_as_tex(macro_metrics, "QA Model")

    return macro_metrics


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
    
                                                                                        
    
