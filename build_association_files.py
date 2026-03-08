import mowl
mowl.init_jvm("4G")
from mowl.owlapi import OWLAPIAdapter
from mowl.datasets import OWLClasses
import java

import pandas as pd
import click as ck
import os
import gzip
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def load_orthology_map(filename):
    hmd_orthology = pd.read_csv(filename, sep='\t', header=None)
    hmd_orthology.columns = ["Human Gene", "Entrez", "MGI Gene", "MGI ID", "High Level Phenos", "Extra"]
    orthology_map = {}
    for index, row in hmd_orthology.iterrows():
        human_gene = row["Entrez"]
        mgi_id = row["MGI ID"]
        orthology_map[mgi_id] = human_gene
    return orthology_map

def load_symbol_to_entrez_map(filename):
    columns = ["UniProtKB-AC", "UniProtKB-ID", "GeneID", "RefSeq", "GI", "PDB", "GO", "UniRef100", "UniRef90", "UniRef50", "UniParc", "PIR", "NCBI-taxon", "MIM", "UniGene", "PubMed", "EMBL", "EMBL-CDS", "Ensembl", "Ensembl_TRS", "Ensembl_PRO", "Additional PubMed"]
    data = pd.read_csv(filename, sep='\t', header=None, low_memory=False)
    data.columns = columns
    symbol_to_entrez = {}
    for index, row in data.iterrows():
        symbol = row["UniProtKB-AC"]
        entrez = row["GeneID"]
        if pd.notna(entrez) and pd.notna(symbol):
            symbol_to_entrez[symbol] = entrez
    logger.info(f"Symbol to Entrez example: {list(symbol_to_entrez.items())[0]}")
    return symbol_to_entrez

def load_ensembl_to_entrez_map(filename):
    data = pd.read_csv(filename, sep='\t')
    data = data[data["#tax_id"] == 9606]  # Filter for human genes only
    ensembl_to_entrez = {}
    for index, row in data.iterrows():
        organism = row["#tax_id"]
        ensembl = row["Ensembl_gene_identifier"]
        entrez = row["GeneID"]
        if pd.notna(entrez) and pd.notna(ensembl):
            ensembl_to_entrez[ensembl] = entrez
    logger.info(f"Ensembl to Entrez example: {list(ensembl_to_entrez.items())[0]}")
    return ensembl_to_entrez

def load_gene_phenotypes(filename, orthology_map):
    logger.debug(f"Orthology map example: {list(orthology_map.items())[0]}")
    mgi_gene_pheno = pd.read_csv(filename, sep='\t', header=None)
    mgi_gene_pheno.columns = ["AlleleComp", "AlleleSymb", "AlleleID", "GenBack", "MP Phenotype", "PubMedID", "MGI ID", "MGI Genotype ID"]
    ortholog_found = 0
    ortholog_not_found = 0
    gene_phenotypes = []
    gene2phenos = {}
    for index, row in mgi_gene_pheno.iterrows():
        genes = row["MGI ID"]
        phenotype = row["MP Phenotype"]
        assert phenotype.startswith("MP:")
        phenotype = "http://purl.obolibrary.org/obo/" + phenotype.replace(":", "_")

        for gene in genes.split('|'):
            human_ortholog = orthology_map.get(gene, None)
            if human_ortholog is None:
                ortholog_not_found += 1
                continue
            else:
                ortholog_found += 1
            gene = "http://mowl.borg/" + str(human_ortholog)
            gene_phenotypes.append((gene, phenotype))
            if gene not in gene2phenos:
                gene2phenos[gene] = set()
            gene2phenos[gene].add(phenotype)

    gene_phenotypes = list(set(gene_phenotypes))  # Remove duplicates
    logger.info(f"Loaded {len(gene_phenotypes)} gene-phenotype associations from {filename}")
    logger.info(f"Orthologous found: {ortholog_found}. Not found: {ortholog_not_found}")
    return gene_phenotypes, gene2phenos

def load_gene_functions(filename, symbol_to_entrez):
    gene_function_pairs = []
    gene2functions = {}
    genes_found = 0
    genes_not_found = 0
    
    with gzip.open(filename, 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            gene, relation, term = parts[1], parts[3], parts[4]
            gene = symbol_to_entrez.get(gene, None)
            if gene is None:
                genes_not_found += 1
                continue
            else:
                genes_found += 1
            gene = "http://mowl.borg/" + str(gene).replace(":", "_")
            term = "http://purl.obolibrary.org/obo/" + str(term).replace(":", "_")
            if relation.startswith('NOT'):
                continue
            
            if gene not in gene2functions:
                gene2functions[gene] = set()
            gene2functions[gene].add(term)
            gene_function_pairs.append((gene, term))

    logger.info(f"Genes mapped to function: {genes_found}. Not found: {genes_not_found}")
    gene_function_pairs = list(set(gene_function_pairs))  # Remove duplicates
    return gene_function_pairs, gene2functions

def load_gene_site(filename, ensembl_to_entrez, threshold):
    gene_site = pd.read_csv(filename, sep='\t', skiprows=4)
    tissue_names = gene_site.columns[2:]  # Skip "Gene ID" and "Gene Name"

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    df = manager.getOWLDataFactory()
    ont = manager.loadOntologyFromOntologyDocument(java.io.File("data/uberon.owl"))
    rdfs_label = df.getRDFSLabel()

    # Build a map from lowercased label to UBERON IRI
    label_to_iri = {}
    for owl_class in ont.getClassesInSignature():
        for axiom in ont.getAnnotationAssertionAxioms(owl_class.getIRI()):
            if axiom.getProperty().equals(rdfs_label):
                if axiom.getValue().isLiteral():
                    label = str(axiom.getValue().asLiteral().get().getLiteral())
                    label_to_iri[label.lower()] = owl_class.getIRI().toString()

    # Map tissue names to UBERON IRIs
    tissue_to_uberon = {}
    unmapped = []
    for tissue in tissue_names:
        uberon_iri = label_to_iri.get(tissue.lower())
        if uberon_iri is not None:
            tissue_to_uberon[tissue] = uberon_iri
        else:
            unmapped.append(tissue)

    logger.info(f"Mapped {len(tissue_to_uberon)}/{len(tissue_names)} tissues to UBERON identifiers")
    if unmapped:
        logger.warning(f"Unmapped tissues: {unmapped}")

    # Build gene-site pairs using only mapped tissues
    gene_site_pairs = []
    gene2sites = {}
    genes_found = 0
    genes_not_found = 0
    for _, row in gene_site.iterrows():
        gene = row["Gene ID"]
        gene = ensembl_to_entrez.get(gene, None)
        if gene is None:
            genes_not_found += 1
            continue
        else:
            genes_found += 1
            
        gene = "http://mowl.borg/" + str(gene).replace(":", "_")
        for tissue, uberon_iri in tissue_to_uberon.items():
            tpm = row[tissue]
            if pd.notna(tpm) and float(tpm) > threshold:
                gene_site_pairs.append((gene, uberon_iri))
                if gene not in gene2sites:
                    gene2sites[gene] = set()
                gene2sites[gene].add(uberon_iri)

    gene_site_pairs = list(set(gene_site_pairs))  # Remove duplicates
    logger.info(f"Loaded {len(gene_site_pairs)} gene-site associations from {filename}")
    logger.info(f"Genes mapped to site: {genes_found}. Not found: {genes_not_found}")
    return gene_site_pairs, gene2sites


def load_disease_phenotypes(filename):
    hpoa = pd.read_csv(filename, sep='\t', comment='#', low_memory=False)

    disease_phenotypes = []
    disease2phenos = {}
    for index, row in hpoa.iterrows():
        disease = row["database_id"]
        phenotype = row["hpo_id"]

        if pd.isna(disease) or pd.isna(phenotype):
            continue
        if not disease.startswith("OMIM:"):
            continue
        assert phenotype.startswith("HP:")
        disease = "http://mowl.borg/" + disease.replace(":", "_")
        phenotype = "http://purl.obolibrary.org/obo/" + phenotype.replace(":", "_")
        disease_phenotypes.append((disease, phenotype))
        if not disease in disease2phenos:
            disease2phenos[disease] = set()
        disease2phenos[disease].add(phenotype)
        
    disease_phenotypes = list(set(disease_phenotypes))  # Remove duplicates
    return disease_phenotypes, disease2phenos



def load_gene_disease(filename):
    hpo_gene_diseases = pd.read_csv(filename, sep='\t')
    hpo_gene_diseases.columns = ["ncbi_gene_id", "gene_symbol", "association_type", "disease_id", "source"]
    gene_disease = []
    for index, row in hpo_gene_diseases.iterrows():
        gene = row["ncbi_gene_id"].split(":")[1]
        disease = row["disease_id"]
        gene = "http://mowl.borg/" + str(gene)
        disease = "http://mowl.borg/" + disease.replace(":", "_")
        gene_disease.append((gene, disease))

    gene_disease = list(set(gene_disease))  # Remove duplicates
    return gene_disease


@ck.command()
@ck.option(
    '--root_dir', '-r', default='data', help='Directory where the data is stored')
def main(root_dir): 
    orthology_map = load_orthology_map(os.path.join(root_dir, 'HMD_HumanPhenotype.rpt'))
    ensembl_to_entrez = load_ensembl_to_entrez_map(os.path.join(root_dir, 'gene2ensembl.gz'))
    symbol_to_entrez = load_symbol_to_entrez_map(os.path.join(root_dir, 'HUMAN_9606_idmapping_selected.tab.gz'))

    
    logger.info("Obtaining Gene-Phenotype associations from MGI_GenePheno.rpt. Genes are represented as MGI IDs and Phenotypes are represented as MP IDs")

    
    gene_pheno_pairs_file = os.path.join(root_dir, 'gene_phenotypes.csv')

    gene_pheno_pairs, gene2phenos = load_gene_phenotypes(os.path.join(root_dir, 'MGI_GenePheno.rpt'), orthology_map)
    logger.info(f"Number of genes: {len(gene2phenos)}. Gene-Phenotype associations: {len(gene_pheno_pairs)}")
    logger.info(f"\tE.g. {gene_pheno_pairs[0]}")

    with open(gene_pheno_pairs_file, 'w') as f:
        f.write("Gene,Phenotype\n")
        for gene, phenotype in gene_pheno_pairs:
            f.write(f"{gene},{phenotype}\n")
    
    logger.info("Obtaining Disease-Phenotype associations from phenotype.hpoa")
    disease_pheno_pairs_file = os.path.join(root_dir, 'disease_phenotypes.csv')

    
    disease_pheno_pairs, disease2phenos = load_disease_phenotypes(os.path.join(root_dir, 'phenotype.hpoa'))
    logger.info(f"Number of diseases: {len(disease2phenos)}. Disease-Phenotype associations: {len(disease_pheno_pairs)}")
    logger.info(f"\tE.g. {disease_pheno_pairs[0]}")

    with open(disease_pheno_pairs_file, 'w') as f:
        f.write("Disease,Phenotype\n")
        for disease, phenotype in disease_pheno_pairs:
            f.write(f"{disease},{phenotype}\n")
    
    
    logger.info("Obtaining Gene-Function associations from goa_human.gaf.gz. Genes are represented as MGI IDs and Functions are represented as GO IDs")
    gene_function_pairs_file = os.path.join(root_dir, 'gene_functions.csv')

    gene_function_pairs, gene2functions = load_gene_functions(os.path.join(root_dir, 'goa_human.gaf.gz'), symbol_to_entrez)
    logger.info(f"Gene-Function associations: {len(gene_function_pairs)}")
    logger.info(f"\tE.g.: {gene_function_pairs[0]}")

    with open(gene_function_pairs_file, 'w') as f:
        f.write("Gene,Function\n")
        for gene, function in gene_function_pairs:
            f.write(f"{gene},{function}\n")


    gene_site_pairs_file = os.path.join(root_dir, 'gene_site.csv')
    gene_site_pairs, gene2sites = load_gene_site(os.path.join(root_dir, 'tpmss.tsv'), ensembl_to_entrez, threshold=4)
    logger.info(f"Gene-Site associations: {len(gene_site_pairs)}")
    logger.info(f"\tE.g.: {gene_site_pairs[0]}")

    with open(gene_site_pairs_file, 'w') as f:
        f.write("Gene,Tissue\n")
        for gene, tissue in gene_site_pairs:
            f.write(f"{gene},{tissue}\n")
            
    genes_with_phenotypes = set([g.split("/")[-1] for g in gene2phenos.keys()])
    genes_with_functions = set([g.split("/")[-1] for g in gene2functions.keys()])
    genes_with_sites = set([g.split("/")[-1] for g in gene2sites.keys()])

    genes_with_info = genes_with_phenotypes.union(genes_with_functions).union(genes_with_sites)
    
    logger.info(f"Number of genes with phenotypes: {len(genes_with_phenotypes)}")
    logger.info(f"Number of genes with functions: {len(genes_with_functions)}")
    logger.info(f"Number of genes with sites: {len(genes_with_sites)}")
    entrez_ids = set(symbol_to_entrez.values())
    genepheno_intersection = entrez_ids.intersection(genes_with_phenotypes)
    genefunction_intersection = entrez_ids.intersection(genes_with_functions)
    genesite_intersection = entrez_ids.intersection(genes_with_sites)

    logger.info(f"Number of genes with phenotypes in symbol to entrez map: {len(genepheno_intersection)}")
    logger.info(f"Number of genes with functions in symbol to entrez map: {len(genefunction_intersection)}")
    logger.info(f"Number of genes with sites in symbol to entrez map: {len(genesite_intersection)}")

    
    logger.info("Constructing Pandas dataframe with with columns: Disease, Gene")
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    ont = manager.loadOntologyFromOntologyDocument(java.io.File(os.path.join(root_dir, 'upheno.owl')))
    classes = set(OWLClasses(ont.getClassesInSignature()).as_str)

    existing_mp_phenotypes = set()
    existing_hp_phenotypes = set()
    for cls in classes:
        if "MP_" in cls:
            existing_mp_phenotypes.add(cls)
        elif "HP_" in cls:
            existing_hp_phenotypes.add(cls)
    logger.info(f"Existing MP phenotypes in ontology: {len(existing_mp_phenotypes)}")
    logger.info(f"Existing HP phenotypes in ontology: {len(existing_hp_phenotypes)}")


    logger.info("Obtaining Gene-Disease associations HPO database")
    gene_disease_pairs_file = os.path.join(root_dir, 'gene_diseases.csv')

    gene_disease = load_gene_disease(os.path.join(root_dir, 'genes_to_disease.txt'))
    logger.info(f"Gene-Disease associations: {len(gene_disease)}")
    logger.info(f"\tE.g.: {gene_disease[0]}")

    genes = []
    diseases = []
    ignored_genes = 0
    logger.info(f"Initial number of gene--disease assocations: {len(gene_disease)}")
    for gene, disease in gene_disease:
        if gene.split("/")[-1] not in genes_with_phenotypes:
            ignored_genes += 1
            continue
        disease_phenos_initial = disease2phenos.get(disease, set())
        disease_phenos = list(disease_phenos_initial.intersection(existing_hp_phenotypes))
        gene_phenos = gene2phenos[gene]
        gene_phenos = list(gene_phenos.intersection(existing_mp_phenotypes))
        assert len(gene_phenos) > 0, f"Gene {gene} has no phenotypes"
        if len(disease_phenos) == 0:
            continue
        genes.append(gene)
        diseases.append(disease)

    logger.info(f"Ignored {ignored_genes} gene-disease associations due to missing gene information")
    logger.info(f"Number of gene-disease pairs after filtering for phenotypes: {len(genes)}")

    with open(gene_disease_pairs_file, 'w') as f:
        used_genes = 0
        f.write("Gene,Disease\n")
        for gene, disease in zip(genes, diseases):
            if gene.split("/")[-1] not in genes_with_phenotypes:
                continue
            used_genes += 1
            f.write(f"{gene},{disease}\n")

        
if __name__ == '__main__':
    main()
