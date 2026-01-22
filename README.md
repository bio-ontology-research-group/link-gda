# multihop-gda


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
