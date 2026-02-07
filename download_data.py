import click as ck
import logging
import os
import wget
import subprocess

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@ck.command()
@ck.option(
    '--save_dir', '-s', default='data', help='Directory to save the data')
def main(save_dir):

    out_dir = os.path.abspath(save_dir)
    logger.info(f'Saving data to {out_dir}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info("Checking if the data is already downloaded")

    if not os.path.exists(os.path.join(out_dir, 'upheno.owl')):
        logger.error("File upheno.owl not found. Downloading it...")
        wget.download("https://purl.obolibrary.org/obo/upheno/v2/upheno.owl", out=out_dir)
        logger.info("Downloaded upheno.owl")
    else:
        logger.info("upheno.owl already exists. Skipping download.")

    if not os.path.exists(os.path.join(out_dir, 'go.owl')):
        logger.error("File go.owl not found. Downloading it...")
        cmd = ["wget", "https://purl.obolibrary.org/obo/go.owl", "-P", out_dir]
        subprocess.run(cmd, check=True)
        logger.info("Downloaded go.owl")
    else:
        logger.info("go.owl already exists. Skipping download.")
        
    if not os.path.exists(os.path.join(out_dir, 'MGI_GenePheno.rpt')):
        logger.error("File MGI_GenePheno.rpt not found. Downloading it for Gene-Phenotype associations")
        wget.download("https://www.informatics.jax.org/downloads/reports/MGI_GenePheno.rpt", out=out_dir)
        logger.info("Downloaded MGI_GenePheno.rpt")
    else:
        logger.info("MGI_GenePheno.rpt already exists. Skipping download.")

    if not os.path.exists(os.path.join(out_dir, 'phenotype.hpoa')):
        logger.error("File phenotype.hpoa not found. Downloading it for Disease-Phenotype associations")
        wget.download("http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa", out=out_dir)
        logger.info("Downloaded phenotype.hpoa")
    else:
        logger.info("phenotype.hpoa already exists. Skipping download.")
        
    if not os.path.exists(os.path.join(out_dir, 'MGI_Geno_DiseaseDO.rpt')):
        logger.error("File MGI_Geno_DiseaseDO.rpt not found. Downloading it for Gene-Disease associations")
        wget.download("https://www.informatics.jax.org/downloads/reports/MGI_Geno_DiseaseDO.rpt", out=out_dir)
        logger.info("Downloaded MGI_Geno_DiseaseDO.rpt")
    else:
        logger.info("MGI_Geno_DiseaseDO.rpt already exists. Skipping download.")

    if not os.path.exists(os.path.join(out_dir, 'mgi.gaf.gz')):
        cmd = ["wget", "https://current.geneontology.org/annotations/mgi.gaf.gz", "-P", out_dir]
        subprocess.run(cmd, check=True)
        logger.info("Downloaded mgi.gaf.gz")
    else:
        logger.info("mgi.gaf.gz already exists. Skipping download.")
        
if __name__ == '__main__':
    main()    
