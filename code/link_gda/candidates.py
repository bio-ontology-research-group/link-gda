"""Annotation-defined candidate pools, independent of evaluation associations."""
from pathlib import Path
import pandas as pd

SOURCES = {'benchmark':'gene_diseases.csv', 'phenotype':'gene_phenotypes.csv',
           'function':'gene_functions.csv'}

def select_candidates(data_dir, pool, available_entities=None, required_genes=()):
    if pool not in SOURCES:
        raise ValueError(f'Unknown candidate pool: {pool}')
    data_dir = Path(data_dir)
    annotations = pd.read_csv(data_dir/SOURCES[pool], dtype=str)
    if pool == 'phenotype':
        # Define eligibility before looking at associations or fold membership.
        edges = pd.read_csv(data_dir/'upheno_edges.tsv', sep='\t', header=None, dtype=str)
        targets = set(edges[0]) | set(edges[2])
        annotations = annotations[annotations.Phenotype.isin(targets)]
    genes = sorted(set(annotations['Gene']))
    if not genes:
        raise ValueError('Candidate pool is empty')
    missing_truth = set(required_genes)-set(genes)
    if missing_truth:
        raise ValueError(f'{len(missing_truth)} evaluated genes absent from {pool} pool')
    if available_entities is not None:
        absent = set(genes)-set(available_entities)
        if absent:
            raise ValueError(f'{len(absent)} {pool} candidates absent from training graph: {sorted(absent)[:5]}')
    return genes

def write_candidates(path, genes):
    if len(genes) != len(set(genes)):
        raise ValueError('Duplicate candidate identifiers')
    Path(path).write_text(''.join(str(g)+'\n' for g in genes))
