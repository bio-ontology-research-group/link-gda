import click as ck
import logging
import os
import subprocess


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

OUT_DIR = os.path.abspath("data")

def download_file(url):
    filename = os.path.basename(url)
    output_path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(output_path):
        logger.info(f"Downloading {filename}...")
        cmd = ["wget", url, "-P", OUT_DIR]
        subprocess.run(cmd, check=True)
        logger.info(f"Downloaded {filename}")
    else:
        logger.info(f"{filename} already exists. Skipping download.")

def main():
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    logger.info("Checking if the data is already downloaded")
    download_file("https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2ensembl.gz")
    download_file("https://purl.obolibrary.org/obo/upheno/v2/upheno.owl")

    download_file("https://purl.obolibrary.org/obo/go.owl")
    download_file("https://purl.obolibrary.org/obo/go/extensions/go-plus.owl")
    download_file("http://purl.obolibrary.org/obo/uberon/releases/2025-12-04/uberon.owl")
    download_file("https://purl.obolibrary.org/obo/mp.owl")
    download_file("https://purl.obolibrary.org/obo/hp.owl")
    download_file("http://purl.obolibrary.org/obo/pato.owl")

    
    download_file("https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz")
    download_file("https://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt") # orthology map
    download_file("https://www.informatics.jax.org/downloads/reports/MGI_GenePheno.rpt") # gene--phenotype annotations
    download_file("https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-01-08/phenotype.hpoa") # disease--phenotype annotations
    download_file("https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-01-08/genes_to_disease.txt") # gene--disease annotations
    download_file("https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz") # gene--functions
    download_file("https://www.ebi.ac.uk/gxa/experiments-content/E-GTEX-8/resources/ExperimentDownloadSupplier.RnaSeqBaseline/tpmss.tsv") # gene--expression
        
if __name__ == '__main__':
    main()    
