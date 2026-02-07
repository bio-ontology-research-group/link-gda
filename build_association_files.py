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
logger.setLevel(logging.INFO)

def load_gene_phenotypes(filename):
            
    mgi_gene_pheno = pd.read_csv(filename, sep='\t', header=None)
    mgi_gene_pheno.columns = ["AlleleComp", "AlleleSymb", "AlleleID", "GenBack", "MP Phenotype", "PubMedID", "MGI ID", "MGI Genotype ID"]

    gene_phenotypes = []
    gene2phenos = {}
    for index, row in mgi_gene_pheno.iterrows():
        genes = row["MGI ID"]
        phenotype = row["MP Phenotype"]
        assert phenotype.startswith("MP:")
        phenotype = "http://purl.obolibrary.org/obo/" + phenotype.replace(":", "_")

        for gene in genes.split('|'):
            gene = "http://mowl.borg/" + str(gene).replace(":", "_")
            gene_phenotypes.append((gene, phenotype))
            if gene not in gene2phenos:
                gene2phenos[gene] = set()
            gene2phenos[gene].add(phenotype)

    gene_phenotypes = list(set(gene_phenotypes))  # Remove duplicates
    return gene_phenotypes, gene2phenos

def load_gene_functions(filename):
    gene_function_pairs = []
    gene2functions = {}
    with gzip.open(filename, 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            gene, relation, term = parts[1], parts[3], parts[4]
            gene = "http://mowl.borg/" + str(gene).replace(":", "_")
            term = "http://purl.obolibrary.org/obo/" + str(term).replace(":", "_")
            if relation.startswith('NOT'):
                continue
            
            if gene not in gene2functions:
                gene2functions[gene] = set()
            gene2functions[gene].add(term)
            gene_function_pairs.append((gene, term))

    gene_function_pairs = list(set(gene_function_pairs))  # Remove duplicates
    return gene_function_pairs, gene2functions


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
    mgi_geno_diseasedo = pd.read_csv(filename, sep='\t')
    mgi_geno_diseasedo.columns = ["AlleleComp", "AlleleSymb", "AlleleID", "GenBack", "MP Phenotype", "PubMedID", "MGI ID Marker", "DO ID", "MIM ID", "MGI Genotype ID"]
    gene_disease = []
    for index, row in mgi_geno_diseasedo.iterrows():
        genes = row["MGI ID Marker"]
        diseases = row["MIM ID"]
        if pd.isna(genes):
            logger.warning(f"Genes not found for {row}")
            continue
        if pd.isna(diseases):
            # logger.warning(f"Disease not found for {row}")
            continue
        assert diseases.startswith("OMIM:")
        
        genes = genes.split("|")
        assert len(genes) == 1, f"Expected only one gene per row, but found {len(genes)} genes in row {index}"
        gene = genes[0]
        
        diseases = diseases.split("|")
        gene = "http://mowl.borg/" + str(gene).replace(":", "_")
        for disease in diseases:
            assert disease.startswith("OMIM:")
            disease = "http://mowl.borg/" + disease.replace(":", "_")
            gene_disease.append((gene, disease))

    gene_disease = list(set(gene_disease))  # Remove duplicates
    return gene_disease

@ck.command()
@ck.option(
    '--root_dir', '-r', default='data', help='Directory where the data is stored')
def main(root_dir): 
       
    logger.info("Obtaining Gene-Phenotype associations from MGI_GenePheno.rpt. Genes are represented as MGI IDs and Phenotypes are represented as MP IDs")

    gene_pheno_pairs_file = os.path.join(root_dir, 'gene_phenotypes.csv')

    gene_pheno_pairs, gene2phenos = load_gene_phenotypes(os.path.join(root_dir, 'MGI_GenePheno.rpt'))
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
    
    logger.info("Obtaining Gene-Disease associations from MGI_Geno_DiseaseDO.rpt. Genes are represented as MGI IDs and Diseases are represented as OMIM IDs")
    gene_disease_pairs_file = os.path.join(root_dir, 'gene_diseases.csv')

    gene_disease = load_gene_disease(os.path.join(root_dir, 'MGI_Geno_DiseaseDO.rpt'))
    logger.info(f"Gene-Disease associations: {len(gene_disease)}")
    logger.info(f"\tE.g.: {gene_disease[0]}")

    with open(gene_disease_pairs_file, 'w') as f:
        f.write("Gene,Disease\n")
        for gene, disease in gene_disease:
            f.write(f"{gene},{disease}\n")

    logger.info("Obtaining Gene-Function associations from mgi.gaf.gz. Genes are represented as MGI IDs and Functions are represented as GO IDs")
    gene_function_pairs_file = os.path.join(root_dir, 'gene_functions.csv')

    gene_function_pairs, gene2functions = load_gene_functions(os.path.join(root_dir, 'mgi.gaf.gz'))
    logger.info(f"Gene-Function associations: {len(gene_function_pairs)}")
    logger.info(f"\tE.g.: {gene_function_pairs[0]}")

    with open(gene_function_pairs_file, 'w') as f:
        f.write("Gene,Function\n")
        for gene, function in gene_function_pairs:
            f.write(f"{gene},{function}\n")

    logger.info("Constructing Pandas dataframe with with columns: Disease, Gene, Disease Phenotypes, Gene Phenotypes, Gene Functions")
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

    genes = []
    gene_filtered_phenos = []
    gene_filtered_functions = []
    diseases = []
    disease_filtered_phenos = []
    diseases_with_no_phenos = 0
    
    for gene, disease in gene_disease:
        disease_phenos = disease2phenos.get(disease, set())
        disease_phenos = list(disease_phenos.intersection(existing_hp_phenotypes))
        gene_phenos = gene2phenos.get(gene, set())
        gene_phenos = list(gene_phenos.intersection(existing_mp_phenotypes))
        if len(disease_phenos) == 0:
            # logger.warning(f"No phenotypes found for disease {disease}. Skipping gene-disease pair ({gene}, {disease})")
            diseases_with_no_phenos += 1
            continue
        genes.append(gene)
        diseases.append(disease)
        gene_filtered_phenos.append(gene_phenos)
        gene_filtered_functions.append(list(gene2functions.get(gene, set())))
        disease_filtered_phenos.append(disease_phenos)

    logger.info(f"Number of gene-disease pairs after filtering for phenotypes: {len(genes)}")
        
    df = pd.DataFrame({
        "Disease": diseases,
        "Gene": genes,
        "Disease Phenotypes": disease_filtered_phenos,
        "Gene Phenotypes": gene_filtered_phenos,
        "Gene Functions": gene_filtered_functions
    })

    df.to_pickle(os.path.join(root_dir, 'gene_disease_associations.pkl'))
    
if __name__ == '__main__':
    main()
