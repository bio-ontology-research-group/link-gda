# multihop-gda

# Dependencies

- Scala 2.11.12 (to align with mOWL)

# Data processing

1. Download data: `python download_data.py`
2. Build association files: `python build_association_files.py`
3. Generate folds: `python generate_folds.py`


## First test: Transductive case with TransE

Command used: 
```
WANDB_MODE=disabled python kge_transe.py --fold 0  --mode transductive --graph4 --no_sweep --only_test
```


Operations:

1. gene_reconstruction: `gene_pheno_embedding - has_pheno_embedding`
2. disease_reconstruction: `disease_pheno_embedding - has_symptom_embedding`


### Original INDIGENA

| Method | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@100 | AUC |
|--------|-----|------|--------|--------|---------|----------|------|
| Inductive BMA | 358.622 | 0.053 | 0.027 | 0.041 | 0.086 | 0.392 | 0.768 |
| Inductive BMM | 389.113 | 0.038 | 0.018 | 0.027 | 0.063 | 0.324 | 0.748 |

### With gene_reconstruction and disease_reconstruction

| Method | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@100 | AUC |
|--------|-----|------|--------|--------|---------|----------|------|
| Inductive BMA | 575.392 | 0.007 | 0.000 | 0.005 | 0.005 | 0.117 | 0.626 |
| Inductive BMM | 589.802 | 0.006 | 0.000 | 0.000 | 0.005 | 0.122 | 0.617 |

### With disease reconstruction only

| Method | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@100 | AUC |
|--------|-----|------|--------|--------|---------|----------|------|
| Inductive BMA | 582.793 | 0.010 | 0.005 | 0.005 | 0.009 | 0.126 | 0.621 |
| Inductive BMM | 603.252 | 0.006 | 0.000 | 0.000 | 0.005 | 0.117 | 0.608 |

### With gene_reconstruction only

| Method | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@100 | AUC |
|--------|-----|------|--------|--------|---------|----------|------|
| Inductive BMA | 350.473 | 0.057 | 0.027 | 0.059 | 0.117 | 0.414 | 0.773 |
| Inductive BMM | 339.297 | 0.041 | 0.009 | 0.036 | 0.090 | 0.392 | 0.780 |




## Data

```
Obtaining Gene-Phenotype associations from MGI_GenePheno.rpt. Genes are represented as MGI IDs and Phenotypes are represented as MP IDs
Loaded 213988 gene-phenotype associations from data/MGI_GenePheno.rpt
Number of genes: 13626. Gene-Phenotype associations: 213988
	E.g. ('http://mowl.borg/IRAG1', 'http://purl.obolibrary.org/obo/MP_0002083')


Obtaining Disease-Phenotype associations from phenotype.hpoa
Number of diseases: 8573. Disease-Phenotype associations: 164006
	E.g. ('http://mowl.borg/OMIM_619482', 'http://purl.obolibrary.org/obo/HP_0002020')


Obtaining Gene-Disease associations from MGI_Geno_DiseaseDO.rpt. Genes are represented as MGI IDs and Diseases are represented as OMIM IDs
Gene-Disease associations: 3363
	E.g.: ('http://mowl.borg/RC3H1', 'http://mowl.borg/OMIM_613145')


Obtaining Gene-Function associations from mgi.gaf.gz. Genes are represented as MGI IDs and Functions are represented as GO IDs
Gene-Function associations: 321532
	E.g.: ('http://mowl.borg/NOX1', 'http://purl.obolibrary.org/obo/GO_0106292')


Mapped 51/53 tissues to UBERON identifiers
Unmapped tissues: ['hippocampus', 'skin']
Loaded 576060 gene-expression associations from data/tpmss.tsv
Gene-Expression associations: 576060
	E.g.: ('http://mowl.borg/SUPT5H', 'http://purl.obolibrary.org/obo/UBERON_0002116')


Constructing Pandas dataframe with with columns: Disease, Gene, Disease Phenotypes, Gene Phenotypes, Gene Functions
Existing MP phenotypes in ontology: 14387
Existing HP phenotypes in ontology: 18546
Number of gene-disease pairs after filtering for phenotypes: 2476

```

# Compiling Scala scripts

```
./compile_projector.sh

```
